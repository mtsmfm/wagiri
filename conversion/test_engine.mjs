// node test_engine.mjs — run RoformerEngine E2E with onnxruntime-node to
// verify the same JS glue as the browser (pack/unpack/stft/istft/chunking)
import { readFileSync } from 'node:fs'
import ort from 'onnxruntime-node'
import { RoformerEngine } from '../src/roformer/engine.js'

// Read test_mix.wav (16-bit PCM stereo 44100Hz).
// The file is an 8s excerpt (2:03-2:11) of "Recondita armonia" (Puccini, Tosca)
// sung by Enrico Caruso, recorded 1909 — public domain. Source:
// https://commons.wikimedia.org/wiki/File:Enrico_Caruso,_Recondita_armonia_(Tosca).ogg
const buf = readFileSync(new URL('./test_mix.wav', import.meta.url))
const dataOffset = buf.indexOf(Buffer.from('data')) + 8
const numSamples = (buf.length - dataOffset) / 4
const left = new Float32Array(numSamples)
const right = new Float32Array(numSamples)
for (let i = 0; i < numSamples; i++) {
  left[i] = buf.readInt16LE(dataOffset + i * 4) / 32768
  right[i] = buf.readInt16LE(dataOffset + i * 4 + 2) / 32768
}
console.log(`loaded ${numSamples} samples`)

const engine = new RoformerEngine({
  ort,
  onProgress: (p) => console.log(`progress ${(p.progress * 100).toFixed(0)}%`),
  executionProviders: ['cpu'],
})
// Node has no Web Worker: run the DSP inline instead of via dsp-worker.js
// (same functions, so numerically identical)
const { stftTorch, istftTorch } = await import('../src/roformer/dsp.js')
engine.dspCall = async (slot, op, a, opts) => {
  if (op === 'stft') {
    const { real, imag } = stftTorch(a, opts.nFft, opts.hop)
    let peak = 1e-9
    for (let i = 0; i < real.length; i++) {
      const x = Math.abs(real[i]); if (x > peak) peak = x
      const y = Math.abs(imag[i]); if (y > peak) peak = y
    }
    return { real, imag, peak }
  }
  const n = a.length / 2
  const real = new Float32Array(n)
  const imag = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    real[i] = a[i * 2] * opts.peak
    imag[i] = a[i * 2 + 1] * opts.peak
  }
  return { out: istftTorch(real, imag, opts.frames, opts.nFft, opts.hop, opts.length) }
}
const model = readFileSync(new URL('../public/models/wagiri-roformer.onnx', import.meta.url))
await engine.loadModel(model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength))

const { vocals, bgm } = await engine.separate(left, right)

const rms = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length)
console.log(`mix RMS: ${rms(left).toFixed(5)}`)
console.log(`vocals RMS: ${rms(vocals.left).toFixed(5)}`)
console.log(`bgm RMS: ${rms(bgm.left).toFixed(5)}`)
if (rms(vocals.left) < 0.01) { console.error('FAIL: vocals is silent'); process.exit(1) }
console.log('ENGINE E2E PASSED')
