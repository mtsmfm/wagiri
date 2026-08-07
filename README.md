# 🍥 wagiri (輪切り)

Audio editing tool that runs entirely in the browser.

- **BGM separation** — runs in-browser on ONNX Runtime Web (WebGPU / WASM); audio is never uploaded anywhere. The model is [smulelabs/windowed-roformer](https://github.com/smulelabs/windowed-roformer) (code and weights both MIT), converted to ONNX by us
- **Waveform trimming** — display the waveform with wavesurfer.js, drag-select a region, export as WAV

## Development

```sh
npm install
npm run dev   # http://localhost:5173 (HOST/PORT env to override)
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for dev tips: verification
scripts, benchmarking in a real browser via CDP, and known gotchas.

The RoFormer models (fp16, plus an fp32 fallback for GPUs without shader-f16)
are fetched from [mtsmfm/windowed-roformer-onnx](https://huggingface.co/mtsmfm/windowed-roformer-onnx)
at runtime and cached by the browser. To use locally generated models instead
(see below), place them in `public/models/` and run with
`VITE_MODEL_BASE=/models`.

## Deployment

Pushing to `main` deploys to GitHub Pages via `.github/workflows/deploy.yml`
(enable Pages with "Source: GitHub Actions" in the repo settings once).
Two things make static hosting work:

- The page needs COOP/COEP headers for multithreaded WASM; GitHub Pages can't
  send headers, so `public/coi-serviceworker.min.js` (MIT) retrofits them via
  a service worker. Hosts that can send real headers (see vite.config.js)
  don't need it — the script is a no-op when the page is already isolated.
- Project pages live under a sub-path, so CI builds with
  `BASE_PATH=/<repo>/`.

## RoFormer model conversion (conversion/)

```sh
python conversion/export_onnx.py \
  --repo <clone of windowed-roformer> \
  --ckpt mbr-win10-sink8.ckpt \
  --out wagiri-roformer.onnx
# Optimizations for onnxruntime-web (WebGPU):
#  1. Bake ORT's graph optimizations (basic) offline (loaded with 'disabled' at runtime)
#  2. Convert Gemm nodes created by MatMulAddFusion back to MatMul+Add
#     (JSEP's Gemm shader is slow)
#  3. Expand Split into Slice (Split falls back to CPU in JSEP; with it gone,
#     every node stays on the GPU and graph capture becomes possible)
#  4. Replace block-local + sink attention with com.microsoft::MultiHeadAttention
#     (paired with the custom WGSL kernel wagiriFusedAttention in the ORT patch)
#  5. Fuse Gelu + remove redundant Expand nodes
#  6. Fuse L2-clip RMSNorm into a single LayerNormalization node
#     (hijacked implementation in the ORT patch)
python conversion/offline_optimize.py wagiri-roformer.onnx opt.onnx
python conversion/gemm_to_matmul.py opt.onnx opt.onnx
python conversion/split_to_slice.py opt.onnx opt.onnx
python conversion/fuse_attention.py opt.onnx opt.onnx
python conversion/fuse_for_jsep.py opt.onnx opt.onnx
python conversion/fuse_rmsnorm.py opt.onnx public/models/wagiri-roformer-fp32.onnx
python conversion/to_fp16.py public/models/wagiri-roformer-fp32.onnx public/models/wagiri-roformer.onnx
node conversion/test_dsp.mjs   # verify the JS-side STFT/iSTFT (requires gen_dsp_testvectors.py)
```

The exporter emits RoPE directly as com.microsoft::RotaryEmbedding instead of
creating a large arithmetic subgraph that has to be folded and fused again.
That operator and the fusion passes (4–6) emit com.microsoft-domain nodes and
**require the patched onnxruntime-web** (they don't work — or mean something
different — on stock ORT).
The patch lives in `patches/onnxruntime-web+*.patch` and is applied to node_modules
by [patch-package](https://github.com/ds300/patch-package) on `npm install`; it
contains 4 self-contained hunks (matmul tile config, the wagiriFusedAttention
WGSL kernel + its MultiHeadAttention gate, and a 2-input LayerNormalization
hijack), all in the JS/WGSL kernel layer — the wasm binaries are stock.
See each conversion script's docstring for details, and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) for what these optimizations buy
(2718ms → ~910ms per chunk on an RTX 3060) and which attempts failed.

export_onnx.py replaces FlexAttention with an equivalent block-local attention and
moves STFT/iSTFT out of the model (to the JS side). It refuses to write output unless
every stage passes a numerical-equivalence check (measured: 1.2e-6 vs FlexAttention,
1.1e-10 vs the original model, 4.3e-8 vs ONNX, relative 1.2e-3 after fp16 conversion).

## License notes

- The code in this repository is MIT (see [LICENSE](LICENSE))
- Uses [onnxruntime-web](https://github.com/microsoft/onnxruntime) (MIT) / [wavesurfer.js](https://github.com/katspaugh/wavesurfer.js) (BSD-3)
- `patches/onnxruntime-web+*.patch` modifies onnxruntime-web's (MIT) JS kernels; to upgrade the runtime, bump the version in package.json, `npm install`, port the patch if it no longer applies, and regenerate it with `npx patch-package onnxruntime-web`
- The RoFormer model weights derive from [smulelabs/windowed-roformer](https://github.com/smulelabs/windowed-roformer) (MIT)
- `conversion/test_mix.wav` is an 8-second excerpt of ["Recondita armonia" sung by Enrico Caruso, recorded 1909](https://commons.wikimedia.org/wiki/File:Enrico_Caruso,_Recondita_armonia_(Tosca).ogg) — public domain
