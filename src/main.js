import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js'
import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom.esm.js'
import * as ort from 'onnxruntime-web'
import { RoformerEngine, ROFORMER } from './roformer/engine.js'
import { t, applyI18n } from './i18n.js'

applyI18n()

ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
// In the production build the wasm runtime ships as plain files under /ort/
// (see ortRuntimeAssets in vite.config.js); in dev, ort resolves them from
// node_modules via import.meta.url
if (import.meta.env.PROD) ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`

const MODEL_CACHE = 'wagiri-model-v5'
for (const old of ['wagiri-model-v1', 'wagiri-model-v2', 'wagiri-model-v3', 'wagiri-model-v4'])
  caches.delete(old).catch(() => {})
// The model has ORT's graph optimizations (constant folding etc.) baked in offline by
// conversion/offline_optimize.py, so runtime optimization is disabled — if enabled,
// MatMulAddFusion fuses MatMul+Add into Gemm, which hits the WebGPU EP's slow Gemm
// shader and runs ~2x slower (see conversion/gemm_to_matmul.py). Split falls back to
// CPU, so it is pre-expanded into Slice by conversion/split_to_slice.py (all nodes stay
// on the GPU = graph capture becomes possible). The model also has fuse_rope /
// fuse_attention / fuse_for_jsep / fuse_rmsnorm applied (paired with the patched
// ORT kernels in patches/onnxruntime-web+*.patch — see the README).
// Models are fetched from Hugging Face by default (and cached in Cache Storage).
// Set VITE_MODEL_BASE=/models to serve locally generated ones from public/models
// (e.g. when iterating on the conversion pipeline).
const MODEL_BASE = import.meta.env.VITE_MODEL_BASE
  || 'https://huggingface.co/mtsmfm/windowed-roformer-onnx/resolve/main/wagiri'
const ROFORMER_MODELS = {
  roformer: `${MODEL_BASE}/wagiri-roformer.onnx`,
  'roformer-fp32': `${MODEL_BASE}/wagiri-roformer-fp32.onnx`,
}
const ROFORMER_SESSION_OPTIONS = navigator.gpu
  ? { graphOptimizationLevel: 'disabled', enableGraphCapture: true, preferredOutputLocation: 'gpu-buffer' }
  : { graphOptimizationLevel: 'disabled' }

const $ = (id) => document.getElementById(id)

let currentFile = null // Blob of the currently loaded audio
let currentAudio = null // { channels: Float32Array[], sampleRate } full-resolution 44.1kHz
let currentName = 'audio'
let ws = null
let regions = null
let activeRegion = null

// ---------- load / dropzone ----------

const dropzone = $('dropzone')
dropzone.addEventListener('click', () => $('file-input').click())
$('file-input').addEventListener('change', (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0])
})
;['dragover', 'dragleave', 'drop'].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault()
    dropzone.classList.toggle('drag', ev === 'dragover')
    if (ev === 'drop' && e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0])
  })
)

async function loadFile(file, name) {
  currentFile = file
  currentName = (name || file.name || 'audio').replace(/\.[^.]+$/, '')
  dropzone.textContent = t('dropzone.loading', { name: name || file.name })
  $('sep-card').style.display = ''
  $('edit-card').style.display = ''
  initWaveSurfer()
  // wavesurfer renders the waveform (decoded at a low rate for display).
  // Decode separately at full resolution (44.1kHz) and keep it for export/separation.
  const [audio] = await Promise.all([decodeFull(file), ws.loadBlob(file)])
  currentAudio = audio
  dropzone.textContent = t('dropzone.loaded', { name: name || file.name })
}

async function decodeFull(blob) {
  const ctx = new AudioContext({ sampleRate: 44100 })
  try {
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer())
    const channels = []
    for (let c = 0; c < buf.numberOfChannels; c++) channels.push(buf.getChannelData(c))
    return { channels, sampleRate: buf.sampleRate }
  } finally {
    ctx.close().catch(() => {})
  }
}

function initWaveSurfer() {
  if (ws) ws.destroy()
  regions = RegionsPlugin.create()
  ws = WaveSurfer.create({
    container: '#waveform',
    height: 140,
    waveColor: '#90a8c8',
    progressColor: '#2b6cb0',
    cursorColor: '#e53e3e',
    minPxPerSec: 20,
    plugins: [regions, ZoomPlugin.create({ scale: 0.4 })],
  })
  regions.enableDragSelection({ color: 'rgba(43,108,176,0.2)' })
  regions.on('region-created', (r) => {
    if (activeRegion && activeRegion !== r) activeRegion.remove()
    activeRegion = r
    updateStatus()
  })
  regions.on('region-updated', updateStatus)
  ws.on('timeupdate', (t) => { $('time').textContent = fmt(t) + ' / ' + fmt(ws.getDuration()) })
  ws.on('ready', () => { $('time').textContent = '0:00.00 / ' + fmt(ws.getDuration()) })
  activeRegion = null
  updateStatus()
}

function fmt(s) {
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`
}

function updateStatus() {
  $('status').textContent = activeRegion
    ? t('status.selection', { start: fmt(activeRegion.start), end: fmt(activeRegion.end), dur: (activeRegion.end - activeRegion.start).toFixed(2) })
    : ''
}

// ---------- playback ----------

$('btn-play').addEventListener('click', () => ws && ws.playPause())
$('btn-play-region').addEventListener('click', () => activeRegion && activeRegion.play())
$('btn-clear').addEventListener('click', () => {
  if (activeRegion) { activeRegion.remove(); activeRegion = null; updateStatus() }
})

// ---------- export selection as WAV ----------

$('btn-export').addEventListener('click', () => {
  if (!currentAudio) return
  if (!activeRegion) { alert(t('alert.selectFirst')); return }
  const { channels, sampleRate } = currentAudio
  const s = Math.max(0, Math.floor(activeRegion.start * sampleRate))
  const e = Math.min(channels[0].length, Math.ceil(activeRegion.end * sampleRate))
  const chans = channels.map((c) => c.subarray(s, e))
  const blob = encodeWav(chans, sampleRate)
  download(blob, `${currentName}_${activeRegion.start.toFixed(2)}-${activeRegion.end.toFixed(2)}.wav`)
})

function download(blob, filename) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}

function encodeWav(chans, sampleRate) {
  const numCh = chans.length, len = chans[0].length
  const bytesPerSample = 2, blockAlign = numCh * bytesPerSample
  const buffer = new ArrayBuffer(44 + len * blockAlign)
  const v = new DataView(buffer)
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  wstr(0, 'RIFF'); v.setUint32(4, 36 + len * blockAlign, true); wstr(8, 'WAVE')
  wstr(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, numCh, true); v.setUint32(24, sampleRate, true)
  v.setUint32(28, sampleRate * blockAlign, true); v.setUint16(32, blockAlign, true)
  v.setUint16(34, 16, true); wstr(36, 'data'); v.setUint32(40, len * blockAlign, true)
  let o = 44
  for (let i = 0; i < len; i++)
    for (let c = 0; c < numCh; c++) {
      const x = Math.max(-1, Math.min(1, chans[c][i]))
      v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true)
      o += 2
    }
  return new Blob([buffer], { type: 'audio/wav' })
}

// ---------- in-browser separation ----------

let roformer = null // initialized engine (lazy)

async function fetchModelCached(url) {
  const cache = await caches.open(MODEL_CACHE).catch(() => null)
  if (cache) {
    const hit = await cache.match(url)
    if (hit) {
      $('sep-status').textContent = t('model.fromCache')
      return hit.arrayBuffer()
    }
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(t('model.fetchFailed', { status: res.status }))
  const total = parseInt(res.headers.get('Content-Length') || '0', 10)
  // Stream a clone directly into Cache Storage. Building a second Response from
  // the completed model would duplicate the entire (roughly 500 MB) buffer.
  const cacheWrite = cache
    ? cache.put(url, res.clone()).catch(() => {})
    : Promise.resolve()
  const reader = res.body.getReader()
  // Content-Length is present on the model response, so fill its final buffer
  // directly instead of retaining every chunk and copying them after download.
  let combined = total ? new Uint8Array(total) : null
  const chunks = [] // Fallback for responses without an accurate Content-Length.
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (combined && loaded + value.length <= combined.length) {
      combined.set(value, loaded)
    } else {
      if (combined) {
        chunks.push(combined.subarray(0, loaded))
        combined = null
      }
      chunks.push(value)
    }
    loaded += value.length
    setProgress(total ? loaded / total : 0, t('model.downloading', { loaded: (loaded / 1e6).toFixed(0), total: total ? ` / ${(total / 1e6).toFixed(0)}MB` : '' }))
  }
  await cacheWrite
  if (!combined) {
    combined = new Uint8Array(loaded)
    let off = 0
    for (const c of chunks) { combined.set(c, off); off += c.length }
  } else if (loaded !== combined.length) {
    combined = combined.slice(0, loaded)
  }
  return combined.buffer
}

function setProgress(value, text) {
  const bar = $('sep-progress')
  bar.style.display = ''
  bar.value = value
  $('sep-status').textContent = text
}

const onProgress = ({ progress, currentSegment, totalSegments }) =>
  setProgress(progress, t('sep.progress', { percent: (progress * 100).toFixed(0), current: currentSegment, total: totalSegments }))

async function getEngine() {
  if (roformer) return roformer
  const eps = ['webgpu', 'wasm']
  let modelUrl = ROFORMER_MODELS.roformer
  // The fp16 model may break WebGPU kernels on devices without shader-f16,
  // so check the adapter's features and auto-fall back to the fp32 model
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter().catch(() => null)
    if (adapter && !adapter.features.has('shader-f16')) {
      modelUrl = ROFORMER_MODELS['roformer-fp32']
      console.log('[wagiri] adapter lacks shader-f16 → falling back to fp32 model')
    }
  }
  console.log(`[wagiri] RoFormer engine, model: ${modelUrl}, executionProviders: ${eps}, crossOriginIsolated: ${crossOriginIsolated}, webgpu: ${!!navigator.gpu}`)
  let engine = new RoformerEngine({ ort, onProgress, executionProviders: eps, sessionOptions: ROFORMER_SESSION_OPTIONS })
  const buf = await fetchModelCached(modelUrl)
  setProgress(1, t('model.init'))
  try {
    await engine.loadModel(buf)
  } catch (err) {
    // Graph capture requires every node to stay on the GPU and some environments
    // reject it. In that case, recreate the engine without capture
    console.warn('[wagiri] graph capture unavailable, falling back:', err.message)
    engine = new RoformerEngine({ ort, onProgress, executionProviders: eps, sessionOptions: { graphOptimizationLevel: 'disabled' } })
    await engine.loadModel(buf)
  }
  roformer = engine
  return engine
}

$('btn-separate').addEventListener('click', async () => {
  if (!currentFile) return
  const btn = $('btn-separate')
  btn.disabled = true
  $('stem-buttons').innerHTML = ''
  try {
    const engine = await getEngine()
    const left = currentAudio.channels[0]
    const right = currentAudio.channels[1] || left
    const { vocals, bgm, sanity } = await engine.separate(left, right)
    $('sep-progress').style.display = 'none'
    if (sanity && sanity.nanCount > 0) {
      $('sep-status').textContent = t('sep.nan', { count: sanity.nanCount })
    } else if (sanity && sanity.mixRms > 0.01 && sanity.vocalsRms < sanity.mixRms * 1e-4) {
      $('sep-status').textContent = t('sep.zeroVocals', { rms: sanity.mixRms.toFixed(3) })
    } else {
      $('sep-status').textContent = t('sep.done')
    }
    addStem(t('stem.vocals'), `${currentName}_vocals.wav`, vocals)
    addStem(t('stem.bgm'), `${currentName}_bgm.wav`, bgm)
  } catch (err) {
    console.error(err)
    $('sep-progress').style.display = 'none'
    $('sep-status').textContent = t('sep.error', { message: err.message })
  } finally {
    btn.disabled = false
  }
})

function addStem(label, filename, { left, right }) {
  const blob = encodeWav([left, right], ROFORMER.SAMPLE_RATE)
  const b = document.createElement('button')
  b.className = 'secondary'
  b.textContent = label
  b.addEventListener('click', () => loadFile(blob, filename))
  $('stem-buttons').appendChild(b)
  const dl = document.createElement('a')
  dl.href = '#'
  dl.textContent = '⬇'
  dl.title = t('stem.downloadTitle', { filename })
  dl.addEventListener('click', (e) => { e.preventDefault(); download(blob, filename) })
  $('stem-buttons').appendChild(dl)
}
