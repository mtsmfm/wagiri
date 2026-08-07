"""Export windowed-roformer (MelBandRoformerWSA) to WebGPU-compatible ONNX.

- STFT/iSTFT live outside the model (implemented in JS) — inputs/outputs are the real-valued STFT representation
- FlexAttention is replaced with block-local attention (mathematically equivalent, static shapes)
- Complex arithmetic and scatter_add are rewritten as real arithmetic and constant matrix products

WebGPU constraint handling (violating these propagates empty tensors at runtime and breaks everything):
- At most 8 storage buffers per shader
  -> per-band Slice instead of a single 60-band Split; Concat as a tree with up to 4 inputs
- Buffer bindings limited to 128MB (spec-guaranteed minimum)
  -> split the Transformer into band/time chunks so every intermediate tensor stays under 128MB
- No Einsum; everything is MatMul

Usage:
  python export_onnx.py --repo /workspace/windowed-roformer \
      --ckpt /workspace/windowed-roformer/mbr-win10-sink8.ckpt \
      --out wagiri-roformer.onnx

Each step has a numerical-equivalence check that raises on failure.
"""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---- Constants (keep consistent with config.yaml / model defaults) ----
SR = 44100
N_FFT = 2048
HOP = 441
WIN = 2048
CHUNK_SECONDS = 8
CHUNK = CHUNK_SECONDS * SR          # 352800
T_FRAMES = CHUNK // HOP + 1         # 801 (center=True)
N_SINKS = 8
HALF_WIN = 5                        # wsa_window_len=10 → ±5
SEQ = N_SINKS + T_FRAMES            # 809
BLOCK = 16                          # block width for block-local attention (>= HALF_WIN+1)
DIM_HEAD = 64
NEG = -1e9
TIME_BAND_CHUNK = 15                # time Transformer: process the band axis 15 at a time
FREQ_TIME_CHUNK = 203               # frequency Transformer: process the time axis 203 at a time
CAT_ARITY = 4                       # max Concat inputs (stays within the 8-binding limit)


def tree_cat(tensors, dim):
    """Tree-shaped concat keeping input count at or below CAT_ARITY"""
    while len(tensors) > 1:
        tensors = [torch.cat(tensors[i:i + CAT_ARITY], dim=dim)
                   for i in range(0, len(tensors), CAT_ARITY)]
    return tensors[0]


def build_local_masks(t_frames):
    """Additive mask constant for block-local attention"""
    n_blocks = math.ceil(t_frames / BLOCK)
    t_pad = n_blocks * BLOCK
    mask = np.full((n_blocks, BLOCK, 3 * BLOCK), NEG, dtype=np.float32)
    for n in range(n_blocks):
        for t in range(BLOCK):
            q_abs = n * BLOCK + t
            if q_abs >= t_frames:
                continue
            for k in range(3 * BLOCK):
                kv_abs = n * BLOCK + (k - BLOCK)
                if 0 <= kv_abs < t_frames and abs(q_abs - kv_abs) <= HALF_WIN:
                    mask[n, t, k] = 0.0
    return n_blocks, t_pad, torch.from_numpy(mask)


class WSABlockAttention(nn.Module):
    """Static-shape implementation equivalent to flex_attention(sliding window + sinks) (MatMul only)"""

    def __init__(self, t_frames):
        super().__init__()
        self.t_frames = t_frames
        n_blocks, t_pad, local_mask = build_local_masks(t_frames)
        self.n_blocks = n_blocks
        self.t_pad = t_pad
        self.register_buffer('local_mask', local_mask, persistent=False)

    def forward(self, q, k, v):
        # q,k,v: (B, H, SEQ, D)
        B, H, S, D = q.shape
        scale = D ** -0.5
        k_sink, k_t = k[:, :, :N_SINKS], k[:, :, N_SINKS:]
        v_sink, v_t = v[:, :, :N_SINKS], v[:, :, N_SINKS:]
        q_sink, q_t = q[:, :, :N_SINKS], q[:, :, N_SINKS:]

        # Sink queries: dense attention over all keys (cheap at 8 x SEQ)
        sim_s = torch.matmul(q_sink, k.transpose(-1, -2)) * scale
        out_sink = torch.matmul(sim_s.softmax(dim=-1), v)

        # Regular queries: block-local + sinks
        pad = self.t_pad - self.t_frames
        q_t = F.pad(q_t, (0, 0, 0, pad))
        k_t = F.pad(k_t, (0, 0, 0, pad))
        v_t = F.pad(v_t, (0, 0, 0, pad))
        nb = self.n_blocks
        q_b = q_t.reshape(B, H, nb, BLOCK, D)
        k_b = k_t.reshape(B, H, nb, BLOCK, D)
        v_b = v_t.reshape(B, H, nb, BLOCK, D)

        def neighbors(x):
            prev = F.pad(x[:, :, :-1], (0, 0, 0, 0, 1, 0))
            nxt = F.pad(x[:, :, 1:], (0, 0, 0, 0, 0, 1))
            return torch.cat([prev, x, nxt], dim=3)

        k_nb = neighbors(k_b)
        v_nb = neighbors(v_b)

        sim_local = torch.matmul(q_b, k_nb.transpose(-1, -2)) * scale
        sim_local = sim_local + self.local_mask
        sim_sink = torch.matmul(q_b, k_sink.transpose(-1, -2).unsqueeze(2)) * scale

        sim = torch.cat([sim_sink, sim_local], dim=-1)
        attn = sim.softmax(dim=-1)
        a_sink, a_local = attn[..., :N_SINKS], attn[..., N_SINKS:]
        out_t = torch.matmul(a_sink, v_sink.unsqueeze(2)) \
              + torch.matmul(a_local, v_nb)
        out_t = out_t.reshape(B, H, self.t_pad, D)[:, :, :self.t_frames]

        return torch.cat([out_sink, out_t], dim=2)


class _ExportRotaryEmbedding(torch.autograd.Function):
    """Run RoPE in PyTorch and export it as ORT's contrib operator."""

    @staticmethod
    def forward(ctx, x, position_ids, cos_cache, sin_cache):
        del ctx, position_ids
        cos = cos_cache.repeat_interleave(2, dim=-1)[None, None]
        sin = sin_cache.repeat_interleave(2, dim=-1)[None, None]
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    @staticmethod
    def symbolic(g, x, position_ids, cos_cache, sin_cache):
        output = g.op(
            'com.microsoft::RotaryEmbedding',
            x, position_ids, cos_cache, sin_cache,
            interleaved_i=1,
        )
        return output.setType(x.type())


class ExportRoPE(nn.Module):
    """RoPE numerically equivalent to rotary_embedding_torch.

    Exporting the arithmetic decomposition caused ORT to fold and then prune
    thousands of block-local constants.  Emit RotaryEmbedding directly; this
    is also the representation consumed by the patched web runtime.
    """

    def __init__(self, rotary_embed, seq_len):
        super().__init__()
        with torch.no_grad():
            device = rotary_embed.freqs.device
            pos = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = rotary_embed.forward(pos).cpu()
        # ORT's interleaved representation stores one value per adjacent pair.
        self.register_buffer('position_ids', torch.arange(seq_len, dtype=torch.int64)[None],
                             persistent=False)
        self.register_buffer('cos', freqs.cos()[:, ::2].contiguous(), persistent=False)
        self.register_buffer('sin', freqs.sin()[:, ::2].contiguous(), persistent=False)

    def forward(self, x):
        position_ids = self.position_ids.expand(x.shape[0], -1)
        return _ExportRotaryEmbedding.apply(
            x, position_ids, self.cos, self.sin)


class PatchedAttention(nn.Module):
    """Same weights and computation as modules.Attention; only attend and rope are swapped"""

    def __init__(self, orig, rope, attend):
        super().__init__()
        self.heads = orig.heads
        self.norm = orig.norm
        self.to_qkv = orig.to_qkv
        self.to_gates = orig.to_gates
        self.to_out = orig.to_out
        self.rope = rope
        self.attend = attend

    def forward(self, x):
        x = self.norm(x)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, DIM_HEAD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        out = self.attend(q, k, v)
        gates = self.to_gates(x)
        out = out * gates.permute(0, 2, 1)[..., None].sigmoid()
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * DIM_HEAD)
        return self.to_out(out)


class DenseAttend(nn.Module):
    """Naive dense attention for the frequency direction (seq=60) (MatMul only)"""

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q, k.transpose(-1, -2)) * scale
        return torch.matmul(sim.softmax(dim=-1), v)


class PatchedTransformer(nn.Module):
    def __init__(self, orig, rope, attend):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([PatchedAttention(attn, rope, attend), ff])
            for attn, ff in orig.layers
        ])
        self.norm = orig.norm

    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class PatchedBandSplit(nn.Module):
    """Same weights as BandSplit; uses per-band Slice instead of a 60-output Split"""

    def __init__(self, orig):
        super().__init__()
        self.dim_inputs = orig.dim_inputs
        self.to_features = orig.to_features

    def forward(self, x):
        outs = []
        start = 0
        for dim_in, net in zip(self.dim_inputs, self.to_features):
            piece = x[..., start:start + dim_in]
            start += dim_in
            outs.append(net(piece).unsqueeze(-2))
        return tree_cat(outs, dim=-2)


class PatchedMaskEstimator(nn.Module):
    """Same weights as MaskEstimator; avoids unbind and 60-input Concat"""

    def __init__(self, orig):
        super().__init__()
        self.to_freqs = orig.to_freqs

    def forward(self, x):
        outs = []
        for i, mlp in enumerate(self.to_freqs):
            outs.append(mlp(x[:, :, i, :]))
        return tree_cat(outs, dim=-1)


class CoreModel(nn.Module):
    """STFT input -> mask-applied STFT output (all real-valued arithmetic).

    Input: stft (1, 2, 1025, T, 2)  [batch, channel, freq, time, re/im]
    Output: masked (1, 2, 1025, T, 2)  vocal STFT
    """

    def __init__(self, m, t_frames=T_FRAMES):
        super().__init__()
        self.t_frames = t_frames
        self.num_bands = m.num_bands
        self.band_split = PatchedBandSplit(m.band_split)
        self.mask_estimator = PatchedMaskEstimator(m.mask_estimators[0])
        self.sink_tokens = m.sink_tokens
        self.register_buffer('freq_indices', m.freq_indices, persistent=False)

        # Fold scatter_add + averaging + DC removal into a single constant matrix (2050 x Fe)
        fe = m.freq_indices.shape[0]
        avg = torch.zeros(2 * 1025, fe)
        denom = m.num_bands_per_freq.repeat_interleave(2).float()
        for e in range(fe):
            f = int(m.freq_indices[e])
            avg[f, e] = 1.0 / float(denom[f])
        avg[0] = 0.0
        avg[1] = 0.0
        self.register_buffer('avg_matrix', avg, persistent=False)

        first_time_attn = m.layers[0][0].layers[0][0]
        first_freq_attn = m.layers[0][1].layers[0][0]
        time_rope = ExportRoPE(first_time_attn.rotary_embed, N_SINKS + t_frames)
        freq_rope = ExportRoPE(first_freq_attn.rotary_embed, self.num_bands)
        wsa = WSABlockAttention(t_frames)
        dense = DenseAttend()

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PatchedTransformer(time_tf, time_rope, wsa),
                PatchedTransformer(freq_tf, freq_rope, dense),
            ])
            for time_tf, freq_tf in m.layers
        ])

    def forward(self, stft_in):
        t = self.t_frames
        seq = N_SINKS + t
        x = stft_in.permute(0, 2, 1, 3, 4).reshape(1, 2 * 1025, t, 2)
        stft_repr = x

        feats = stft_repr[:, self.freq_indices]                 # (1, Fe, T, 2)
        feats = feats.permute(0, 2, 1, 3).reshape(1, t, -1)     # (1, T, Fe*2)
        feats = self.band_split(feats)                          # (1, T, bands, dim)

        sinks = self.sink_tokens[None]
        h = torch.cat([sinks, feats], dim=1)                    # (1, SEQ, bands, dim)

        for time_tf, freq_tf in self.layers:
            b, tt, f, d = h.shape
            # Time Transformer: chunk the band axis to stay within the 128MB limit
            ht = h.permute(0, 2, 1, 3).reshape(f, tt, d)
            ht = tree_cat([time_tf(ht[i:i + TIME_BAND_CHUNK])
                           for i in range(0, f, TIME_BAND_CHUNK)], dim=0)
            h = ht.reshape(1, f, tt, d).permute(0, 2, 1, 3)
            # Frequency Transformer: chunk the time axis
            hf = h.reshape(tt, f, d)
            hf = tree_cat([freq_tf(hf[i:i + FREQ_TIME_CHUNK])
                           for i in range(0, tt, FREQ_TIME_CHUNK)], dim=0)
            h = hf.reshape(1, tt, f, d)

        h = h[:, N_SINKS:]                                      # (1, T, bands, dim)
        mask = self.mask_estimator(h)                           # (1, T, Fe*2)
        mask = mask.reshape(1, t, -1, 2).permute(0, 2, 1, 3)    # (1, Fe, T, 2)

        # Map back to 2050 frequencies via the averaging matrix (linear map, re/im independent) — as MatMul
        fe = mask.shape[1]
        mask2 = mask.reshape(fe, t * 2)
        mr = torch.matmul(self.avg_matrix, mask2).reshape(1, 2 * 1025, t, 2)

        # Complex multiplication (a+bi)(c+di)
        a, b_ = stft_repr[..., 0], stft_repr[..., 1]
        c, d_ = mr[..., 0], mr[..., 1]
        re = a * c - b_ * d_
        im = a * d_ + b_ * c
        out = torch.stack([re, im], dim=-1)
        return out.reshape(1, 1025, 2, t, 2).permute(0, 2, 1, 3, 4)


# ---------------- Verification ----------------

def check(name, a, b, atol):
    d = (a - b).abs().max().item()
    status = 'OK' if d <= atol else 'FAIL'
    print(f'[{status}] {name}: max diff {d:.3e} (atol {atol:.0e})')
    if d > atol:
        raise SystemExit(f'equivalence check failed: {name}')


@torch.no_grad()
def verify_wsa(device):
    from torch.nn.attention.flex_attention import flex_attention
    from flex_attention_utils import generate_sliding_window_with_sinks, create_block_mask_cached
    q = torch.randn(2, 8, SEQ, DIM_HEAD, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask_mod = generate_sliding_window_with_sinks(10, N_SINKS)
    bm = create_block_mask_cached(mask_mod, None, None, SEQ, SEQ, device.type)
    ref = flex_attention(q, k, v, block_mask=bm)
    ours = WSABlockAttention(T_FRAMES).to(device)(q, k, v)
    check('WSA block attention vs flex_attention', ours, ref, 2e-4)


@torch.no_grad()
def verify_rope(model, device):
    attn = model.layers[0][0].layers[0][0]
    rope = ExportRoPE(attn.rotary_embed, SEQ).to(device)
    x = torch.randn(3, 8, SEQ, DIM_HEAD, device=device)
    ref = attn.rotary_embed.rotate_queries_or_keys(x)
    check('RoPE reimplementation', rope(x), ref, 1e-5)


@torch.no_grad()
def verify_core(model, core, device):
    t_frames = core.t_frames
    chunk = (t_frames - 1) * HOP
    audio = torch.randn(1, 2, chunk, device=device) * 0.1
    ref = model(audio)

    window = torch.hann_window(WIN, device=device)
    spec = torch.stft(audio.reshape(2, chunk), n_fft=N_FFT, hop_length=HOP,
                      win_length=WIN, window=window, return_complex=True)
    stft_in = torch.view_as_real(spec)[None]
    masked = core(stft_in)
    spec_out = torch.view_as_complex(masked[0].contiguous())
    recon = torch.istft(spec_out, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                        window=window, length=None)
    ref_cmp = ref[0][..., :recon.shape[-1]]
    check('CoreModel + external STFT/iSTFT vs original model', recon, ref_cmp, 5e-4)


@torch.no_grad()
def verify_onnx(core, out_path):
    import onnxruntime as rt
    x = torch.randn(1, 2, 1025, T_FRAMES, 2) * 0.05
    ref = core.cpu()(x)
    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.enable_mem_pattern = False
    sess = rt.InferenceSession(out_path, opts, providers=['CPUExecutionProvider'])
    got = sess.run(None, {'stft_in': x.numpy()})[0]
    check('ONNX (CPU) vs PyTorch CoreModel', torch.from_numpy(got), ref, 5e-4)


def split_large_concats(out_path, max_inputs=4):
    """Re-split large-input Concats that the exporter re-fused back into a tree (protobuf surgery)"""
    import onnx
    m = onnx.load(out_path)
    g = m.graph
    changed = True
    serial = 0
    while changed:
        changed = False
        for idx, n in enumerate(list(g.node)):
            if n.op_type != 'Concat' or len(n.input) <= max_inputs:
                continue
            axis = n.attribute[0].i
            inputs = list(n.input)
            new_nodes = []
            layer = inputs
            while len(layer) > 1:
                nxt = []
                for i in range(0, len(layer), max_inputs):
                    grp = layer[i:i + max_inputs]
                    if len(grp) == 1:
                        nxt.append(grp[0])
                        continue
                    serial += 1
                    is_last = len(layer) <= max_inputs
                    out_name = n.output[0] if is_last else f'{n.name}_tree_{serial}'
                    new_nodes.append(onnx.helper.make_node(
                        'Concat', grp, [out_name],
                        name=f'{n.name}_tree_{serial}', axis=axis))
                    nxt.append(out_name)
                layer = nxt
            # Replace the original node (insert at the same position to preserve topological order)
            del g.node[idx]
            for j, nn in enumerate(new_nodes):
                g.node.insert(idx + j, nn)
            changed = True
            break
    onnx.save(m, out_path)
    print(f'[OK] split_large_concats: rewrote large Concat nodes')


def verify_webgpu_constraints(out_path):
    """Check WebGPU constraints (Split/Concat input/output counts) by graph inspection"""
    import onnx
    m = onnx.load(out_path, load_external_data=False)
    bad = []
    for n in m.graph.node:
        io = 0
        if n.op_type == 'Split':
            io = 1 + len(n.output)
        elif n.op_type == 'Concat':
            io = len(n.input) + 1
        if io > 8:
            bad.append((n.op_type, n.name, io))
        if n.op_type == 'Einsum':
            bad.append(('Einsum', n.name, '-'))
    if bad:
        for b in bad:
            print('  NG:', b)
        raise SystemExit('WebGPU constraint check failed')
    print('[OK] WebGPU constraints: no Einsum, all Split/Concat <= 8 bindings')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='wagiri-roformer.onnx')
    ap.add_argument('--skip-flex-check', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from model import MelBandRoformerWSA
    import modules as wr_modules

    # RMSNorm's F.normalize default eps=1e-12 gets clamped to 1e-7 by fp16 conversion,
    # then flushed to 0 by f16 denormal flushing (GPU), producing 0/0=NaN on silent frames.
    # Raise it to 1e-4, representable as an f16 normal number (no impact outside near-silence).
    def _safe_rmsnorm_forward(self, x):
        return F.normalize(x, dim=-1, eps=1e-4) * self.scale * self.gamma
    wr_modules.RMSNorm.forward = _safe_rmsnorm_forward

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = MelBandRoformerWSA()
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    if not args.skip_flex_check:
        verify_wsa(device)
    verify_rope(model, device)

    # The original model's flex_attention uses compile=True (requires triton). Swap in the uncompiled version for CPU comparison
    from torch.nn.attention.flex_attention import flex_attention as _fa_eager
    for mod in model.modules():
        if hasattr(mod, 'flex_attn_fn'):
            mod.flex_attn_fn = _fa_eager
    model = model.cpu()
    core_small = CoreModel(model, t_frames=101).eval()
    verify_core(model, core_small, torch.device('cpu'))
    core = CoreModel(model, t_frames=T_FRAMES).eval()

    # Guard against OOM during export
    import gc
    del core_small
    for p in core.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    dummy = torch.randn(1, 2, 1025, T_FRAMES, 2) * 0.05
    with torch.no_grad():
        torch.onnx.export(
            core, (dummy,), args.out,
            input_names=['stft_in'], output_names=['masked_stft'],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f'exported: {args.out}')
    split_large_concats(args.out)
    verify_webgpu_constraints(args.out)
    verify_onnx(core, args.out)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
