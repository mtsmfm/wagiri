// node test_dsp.mjs — compare the JS stftTorch/istftTorch against torch output
import { readFileSync } from 'node:fs'
import { stftTorch, istftTorch } from '../src/roformer/dsp.js'

const ref = JSON.parse(readFileSync(new URL('./dsp_testvectors.json', import.meta.url)))
const sig = Float32Array.from(ref.signal)

const { real, imag, numFrames, numBins } = stftTorch(sig, 2048, 441)

if (numFrames !== ref.numFrames || numBins !== ref.numBins) {
  console.error(`FAIL shape: got ${numBins}x${numFrames}, want ${ref.numBins}x${ref.numFrames}`)
  process.exit(1)
}

let maxd = 0
for (let f = 0; f < numBins; f++) {
  for (let t = 0; t < 5; t++) {
    maxd = Math.max(maxd,
      Math.abs(real[f * numFrames + t] - ref.specRealHead[f][t]),
      Math.abs(imag[f * numFrames + t] - ref.specImagHead[f][t]))
  }
}
console.log(`STFT head max diff: ${maxd.toExponential(2)}`)
if (maxd > 2e-3) { console.error('FAIL stft'); process.exit(1) }

let absSum = 0
for (let i = 0; i < real.length; i++) absSum += Math.hypot(real[i], imag[i])
const relErr = Math.abs(absSum - ref.specAbsSum) / ref.specAbsSum
console.log(`STFT abs-sum rel err: ${relErr.toExponential(2)}`)
if (relErr > 1e-4) { console.error('FAIL stft abs sum'); process.exit(1) }

const recon = istftTorch(real, imag, numFrames, 2048, 441, sig.length)
let maxr = 0
for (let i = 0; i < sig.length; i++) {
  maxr = Math.max(maxr, Math.abs(recon[i] - ref.recon[i]))
}
console.log(`iSTFT vs torch.istft max diff: ${maxr.toExponential(2)}`)
if (maxr > 2e-3) { console.error('FAIL istft'); process.exit(1) }

console.log('DSP CHECKS PASSED')
