// Run a benchmark/diagnostic task inside a real browser via CDP.
// The app ships no debug hooks: this script builds its own harness in the page
// by importing the vite-dev-served modules (patched ort.all.mjs + engine.js).
//
// usage:
//   CDP_URL=http://<host>:9222 node tools/bench.mjs <task.mjs> [--keep]
//     --keep: reuse the already-open page and its harness (no reload)
//
// The task file is plain JS evaluated in the page with `ctx` in scope:
//   ctx.ort, ctx.RoformerEngine  — the vite-served modules
//   ctx.getEngine()              — lazily built engine (graph capture when WebGPU)
//   ctx.audio()                  — decoded conversion/test_mix.wav
//   ctx.separate()               — run separation, returns sanity {timings, rms, ...}
// Whatever the task `return`s is printed as JSON.
//
// Example task:
//   const cold = await ctx.separate()
//   const warm = await ctx.separate()
//   return { cold: cold.timings, warm: warm.timings }
//
// See docs/DEVELOPMENT.md for the Chrome-side setup.
import puppeteer from 'puppeteer-core'
import { readFileSync } from 'node:fs'

const APP_URL = process.env.APP_URL || 'http://localhost:5173/'
const AUDIO_URL = '/conversion/test_mix.wav'
const taskFile = process.argv[2]
const keep = process.argv.includes('--keep')
if (!taskFile || !process.env.CDP_URL) {
  console.error('usage: CDP_URL=http://<host>:9222 node tools/bench.mjs <task.mjs> [--keep]')
  process.exit(1)
}
const code = readFileSync(taskFile, 'utf8')

const BOOTSTRAP = `window.__wagiri = await (async () => {
  const ort = await import('/node_modules/onnxruntime-web/dist/ort.all.mjs')
  const { RoformerEngine } = await import('/src/roformer/engine.js')
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
  let engine = null
  const getEngine = async () => {
    if (engine) return engine
    const buf = await (await fetch('/models/wagiri-roformer.onnx')).arrayBuffer()
    const e = new RoformerEngine({
      ort, onProgress: () => {}, executionProviders: ['webgpu', 'wasm'],
      sessionOptions: navigator.gpu
        ? { graphOptimizationLevel: 'disabled', enableGraphCapture: true, preferredOutputLocation: 'gpu-buffer' }
        : { graphOptimizationLevel: 'disabled' },
    })
    await e.loadModel(buf)
    return (engine = e)
  }
  const ac = new AudioContext({ sampleRate: 44100 })
  const ab = await ac.decodeAudioData(await (await fetch(${JSON.stringify(AUDIO_URL)})).arrayBuffer())
  ac.close().catch(() => {})
  const channels = []
  for (let c = 0; c < ab.numberOfChannels; c++) channels.push(ab.getChannelData(c))
  const audioData = { channels, sampleRate: ab.sampleRate }
  return {
    ort, RoformerEngine, getEngine,
    audio: () => audioData,
    separate: async () => {
      const e = await getEngine()
      const l = audioData.channels[0], r = audioData.channels[1] || l
      return (await e.separate(l, r)).sanity
    },
  }
})()`

const browser = await puppeteer.connect({ browserURL: process.env.CDP_URL, defaultViewport: null, protocolTimeout: 1800000 })
try {
  const pages = await browser.pages()
  let page = pages.find((p) => p.url().startsWith(APP_URL))
  if (!page) page = await browser.newPage()
  page.on('console', (m) => console.error('[console]', m.text().slice(0, 300)))
  if (!keep || !page.url().startsWith(APP_URL)) {
    await page.goto(APP_URL, { waitUntil: 'domcontentloaded' })
  }
  // Without this, a stale HTTP cache can serve old modules/models and
  // invalidate the whole measurement
  await page.setCacheEnabled(false).catch(() => {})
  // WebGPU throttles background tabs
  await page.bringToFront().catch(() => {})
  await page.evaluate(`(async () => { if (${!keep} || !window.__wagiri) { ${BOOTSTRAP} } })()`)
  const result = await page.evaluate(
    `(async () => { const ctx = window.__wagiri; ${code}\n })()`
  )
  console.log(JSON.stringify(result))
} finally {
  await browser.disconnect()
}
