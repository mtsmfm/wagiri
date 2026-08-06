import { defineConfig } from 'vite'
import { copyFileSync, mkdirSync, readdirSync } from 'node:fs'

// ort.all.mjs loads its wasm runtime (ort-wasm-simd-threaded*.mjs/.wasm) via
// dynamic import at runtime; the bundler can't see those imports, so ship the
// files as-is under dist/ort/ and point ort.env.wasm.wasmPaths there (main.js)
function ortRuntimeAssets() {
  return {
    name: 'wagiri-ort-runtime-assets',
    closeBundle() {
      const src = 'node_modules/onnxruntime-web/dist'
      mkdirSync('dist/ort', { recursive: true })
      for (const f of readdirSync(src).filter((f) => f.startsWith('ort-wasm-simd-threaded')))
        copyFileSync(`${src}/${f}`, `dist/ort/${f}`)
    },
  }
}

// COOP/COEP: make the page crossOriginIsolated to enable WASM multithreading
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

// Overridable for setups where the browser reaches the dev machine remotely,
// e.g. HOST=0.0.0.0 PORT=18002 npm run dev
const serverConfig = {
  host: process.env.HOST || 'localhost',
  port: Number(process.env.PORT) || 5173,
  // Needed when accessing through a hostname (e.g. a tailnet); harmless on localhost
  allowedHosts: true,
  headers: isolationHeaders,
}

export default defineConfig({
  // Deploys under a sub-path on GitHub Pages (BASE_PATH=/wagiri/ in CI)
  base: process.env.BASE_PATH || '/',
  plugins: [ortRuntimeAssets()],
  resolve: {
    // dist/ort.all.mjs carries our kernel-optimization patches, applied to
    // node_modules by patch-package (patches/onnxruntime-web+*.patch) on npm install.
    // The package's default export is the minified bundle, so alias to the patched file.
    // wasm assets are resolved from the same directory via import.meta.url.
    alias: {
      'onnxruntime-web': new URL('./node_modules/onnxruntime-web/dist/ort.all.mjs', import.meta.url).pathname,
    },
  },
  // onnxruntime-web resolves its own .wasm/.mjs via import.meta.url, so keep it
  // out of pre-bundling (bundling breaks asset resolution in dev)
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: serverConfig,
  preview: serverConfig,
})
