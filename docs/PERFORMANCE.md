# Performance notes

How the RoFormer engine got from 2718ms to ~910ms per 8-second chunk, what the
current numbers are, and which attempts failed (recorded so future work doesn't
retry them blindly). All measurements: Windows / RTX 3060 / Chrome ~150,
via `tools/bench.mjs` (see [DEVELOPMENT.md](DEVELOPMENT.md)).

## Reference numbers

| Runtime | ms / 8s chunk |
|---|---|
| WebGPU fp16, patched kernels, graph capture (this repo, warm) | **~910** |
| ONNX Runtime CUDA EP, fp16, full optimization (same GPU) | 1233 |
| Native PyTorch, autocast fp16, flash attention | 332 |
| WebGPU before this work (stock ORT, fp16 model) | 2718 |

End-to-end (1 chunk incl. STFT/iSTFT in workers): ~1.1s warm; multi-chunk
throughput is better (~1.02s/chunk) because DSP overlaps GPU inference.
The first run after page load compiles all WGSL pipelines (cold ≈ 2x warm).

## What the speedup consists of

- **Offline-baked graph optimization, runtime optimization disabled.** ORT's
  runtime MatMulAddFusion produces Gemm nodes, and JSEP's Gemm shader is ~2x
  slower than MatMul+Add — so optimizations are baked offline
  (conversion/offline_optimize.py) and Gemm is converted back
  (conversion/gemm_to_matmul.py).
- **All nodes GPU-resident.** Split falls back to CPU in JSEP; expanding it to
  Slice (conversion/split_to_slice.py) removes every CPU hop and makes graph
  capture possible (enableGraphCapture + GPU-buffer IO binding in engine.js).
- **Op fusions** (RoPE → RotaryEmbedding, Gelu, L2-clip RMSNorm →
  LayerNormalization) cut ~1,700 dispatches.
- **Custom WGSL attention kernel** (wagiriFusedAttention in the ORT patch):
  sink-8 + sliding-window block-local attention in one dispatch per node,
  online softmax in f32, vec4<f16> dot products, shared-memory padding against
  bank conflicts. Replaced ~56-dispatch attention subgraphs.
- **Matmul tile tuning**: 32x64 tiles (wg [16,8], tileInner 64) measured ~5%
  faster than the default 32x32 on RTX 3060.
- **DSP off the GPU path**: STFT/iSTFT run in 2 workers (L/R) and overlap the
  next/previous chunk's GPU inference (chunk pipelining, numerically
  identical to sequential).

The fusions change fp16 rounding order, so output is not bit-identical to the
pre-fusion reference: at most ~17 LSB in int16 (about -65dB, inaudible).

## Attempts that did not pan out

- **3D attention layout (removing the head-split Gather/Transpose).** Graph
  surgery replacing each attention's Gather×3 + Transpose×5 head split with
  `Slice ×3 → RotaryEmbedding(3D) → MHA(num_heads=8)` worked (48/48 sites,
  3739 → 3307 nodes, numerically verified on CPU), but extending the WGSL
  fused-attention kernel to heads mode hung the GPU (device lost, root cause
  never found — took the browser down twice). Expected gain was only a few
  dispatches per attention, so it was dropped rather than debugged further.
- **WebNN (DirectML).** Tensor cores are reachable — a matmul microbenchmark
  hit 31.6 TFLOPS vs 5.0 on WebGPU — but `MLGraphBuilder.build()` time scales
  pathologically with graph size (1501 nodes > 4 min; the full model never
  finished in > 40 min). Unusable as of Chrome 150.
- **WebGPU tensor cores on Windows.** `chromium-experimental-subgroup-matrix`
  is not implemented on the D3D12 backend (Microsoft removed WaveMatrix from
  DXC; SM6.9 cooperative vectors cover only matrix-vector), there is no flag
  to pick the Vulkan backend on Windows, and WSL's dzn Vulkan maps back to
  D3D12. This is what blocks closing the gap to native PyTorch (332ms/chunk
  with flash attention on tensor cores); ~910ms/chunk is close to the
  fp16-FMA limit without them.
- **Parallelizing phase 2 (softmax) of the fused attention kernel** across
  16 threads produced wrong values (max diff 0.489) and would have saved only
  ~3ms — kept serial.

## Possible upstream contributions

Not done yet, but worth doing: report the MultiHeadAttention `scale`
attribute not being forwarded to the JS kernel (bug), propose the matmul tile
heuristics with benchmark data, and offer the sliding-window+sink attention
as a proper GQA-style kernel.
