# Development tips

## Dev server

```sh
npm run dev                          # vite on localhost:5173
HOST=0.0.0.0 PORT=18002 npm run dev  # e.g. when the browser is on another machine
```

The server sends COOP/COEP headers (see vite.config.js) so the page is
`crossOriginIsolated` — required for multithreaded WASM. Any production host
must send the same two headers.

After changing `vite.config.js` or anything under `patches/`, restart the dev
server and clear vite's transform cache — a stale cache serving old modules
has repeatedly wasted debugging time:

```sh
rm -rf node_modules/.vite
```

## Verification scripts

```sh
node conversion/test_dsp.mjs      # JS STFT/iSTFT vs torch reference vectors
                                  # (needs conversion/gen_dsp_testvectors.py once)
node conversion/test_engine.mjs   # engine E2E on onnxruntime-node (CPU)
```

Note: the CPU E2E run exercises the JS glue (packing, chunking, DSP, sanity),
not the patched WebGPU kernels — ORT's CPU MultiHeadAttention computes plain
full attention, so its output differs from the browser's. Kernel-level
verification has to happen in a browser (below).

## Benchmarking in a real browser (CDP)

WebGPU numbers are only meaningful on real hardware, and this repo was tuned
against a browser driven over the Chrome DevTools Protocol. The app contains
no debug hooks — `tools/bench.mjs` builds its harness inside the page by
importing the vite-served modules directly.

1. Start Chrome with CDP enabled. Since Chrome 136 this requires a
   non-default profile:

   ```sh
   chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-cdp
   ```

   (Add `--enable-unsafe-webgpu --enable-webgpu-developer-features` when
   experimenting with WebGPU features.)

2. Start the dev server, then run a task:

   ```sh
   CDP_URL=http://localhost:9222 node tools/bench.mjs mytask.mjs
   # APP_URL=http://<host>:<port>/ if the dev server is not on localhost:5173
   ```

   The task file is JS evaluated in the page with `ctx` in scope
   (`ctx.separate()`, `ctx.getEngine()`, `ctx.audio()`, `ctx.ort`, …);
   its return value is printed as JSON. The bundled public-domain
   `conversion/test_mix.wav` is preloaded as the input signal.

Gotchas learned the hard way:

- `tools/bench.mjs` disables the HTTP cache on the page — without that, a
  stale cached module or model silently invalidates the measurement.
- Keep the tab foregrounded (the script calls `bringToFront`): WebGPU work in
  background tabs gets throttled.
- The first run after a page load compiles all WGSL pipelines (cold ≈ 2x
  warm); benchmark from the second `ctx.separate()` on.
- If the GPU device is lost (bad kernel experiment), Chrome may need a full
  restart, not just a page reload.

## Performance background

See [PERFORMANCE.md](PERFORMANCE.md) for the optimization breakdown, reference
numbers, and the list of attempts that did not pan out.
