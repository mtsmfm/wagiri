// FFT core is based on fft.js from demucs-web (MIT License, Copyright timcsy).
// STFT/iSTFT are written to numerically match torch.stft / torch.istft
// (center=true, pad_mode=reflect, periodic Hann, normalized=false).

const fftTwiddles = new Map()
const ifftTwiddles = new Map()
const hannWindows = new Map()
const bitReverseTables = new Map()
const windowSumCache = new Map()

function getTwiddles(map, n, sign) {
  if (map.has(n)) return map.get(n)
  const real = new Float32Array(n / 2)
  const imag = new Float32Array(n / 2)
  for (let k = 0; k < n / 2; k++) {
    const angle = sign * 2 * Math.PI * k / n
    real[k] = Math.cos(angle)
    imag[k] = Math.sin(angle)
  }
  const t = { real, imag }
  map.set(n, t)
  return t
}

function getHannWindow(size) {
  if (hannWindows.has(size)) return hannWindows.get(size)
  const w = new Float32Array(size)
  for (let i = 0; i < size; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / size))
  hannWindows.set(size, w)
  return w
}

function getBitReverseTable(n) {
  if (bitReverseTables.has(n)) return bitReverseTables.get(n)
  const bits = Math.log2(n) | 0
  const table = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    let r = 0, x = i
    for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1 }
    table[i] = r
  }
  bitReverseTables.set(n, table)
  return table
}

function fftCore(realOut, imagOut, n, twiddles) {
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2
    const step = n / size
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < half; j++) {
        const k = j * step
        const tR = twiddles.real[k], tI = twiddles.imag[k]
        const i1 = i + j, i2 = i + j + half
        const eR = realOut[i1], eI = imagOut[i1]
        const oR = realOut[i2] * tR - imagOut[i2] * tI
        const oI = realOut[i2] * tI + imagOut[i2] * tR
        realOut[i1] = eR + oR; imagOut[i1] = eI + oI
        realOut[i2] = eR - oR; imagOut[i2] = eI - oI
      }
    }
  }
}

function fft(realOut, imagOut, realIn, n) {
  const table = getBitReverseTable(n)
  for (let i = 0; i < n; i++) {
    realOut[i] = realIn[table[i]]
    imagOut[i] = 0
  }
  fftCore(realOut, imagOut, n, getTwiddles(fftTwiddles, n, -1))
}

function ifft(realOut, imagOut, realIn, imagIn, n) {
  const table = getBitReverseTable(n)
  for (let i = 0; i < n; i++) {
    const j = table[i]
    realOut[i] = realIn[j]
    imagOut[i] = imagIn[j]
  }
  fftCore(realOut, imagOut, n, getTwiddles(ifftTwiddles, n, 1))
  for (let i = 0; i < n; i++) { realOut[i] /= n; imagOut[i] /= n }
}

function reflectPad(signal, pad) {
  const n = signal.length
  const out = new Float32Array(n + 2 * pad)
  out.set(signal, pad)
  for (let i = 0; i < pad; i++) {
    out[i] = signal[pad - i]
    out[pad + n + i] = signal[n - 2 - i]
  }
  return out
}

/**
 * Matches torch.stft(x, n_fft, hop, win_length=n_fft, hann periodic, center=True,
 * pad_mode='reflect', normalized=False, onesided=True).
 * Return value is freq-major: real[f * numFrames + t]
 */
export function stftTorch(signal, nFft, hop) {
  const pad = nFft / 2
  const padded = reflectPad(signal, pad)
  const numFrames = Math.floor((padded.length - nFft) / hop) + 1
  const numBins = nFft / 2 + 1
  const window = getHannWindow(nFft)

  const real = new Float32Array(numBins * numFrames)
  const imag = new Float32Array(numBins * numFrames)
  const fr = new Float32Array(nFft)
  const fi = new Float32Array(nFft)
  const frame = new Float32Array(nFft)

  for (let t = 0; t < numFrames; t++) {
    const start = t * hop
    for (let i = 0; i < nFft; i++) frame[i] = padded[start + i] * window[i]
    fft(fr, fi, frame, nFft)
    for (let f = 0; f < numBins; f++) {
      real[f * numFrames + t] = fr[f]
      imag[f * numFrames + t] = fi[f]
    }
  }
  return { real, imag, numFrames, numBins }
}

/**
 * Matches torch.istft(spec, n_fft, hop, hann periodic, center=True, length).
 * Input is freq-major (same layout as stftTorch).
 */
export function istftTorch(real, imag, numFrames, nFft, hop, length) {
  const numBins = nFft / 2 + 1
  const pad = nFft / 2
  const window = getHannWindow(nFft)
  const totalLength = (numFrames - 1) * hop + nFft

  const output = new Float32Array(totalLength)
  const fullR = new Float32Array(nFft)
  const fullI = new Float32Array(nFft)
  const outR = new Float32Array(nFft)
  const outI = new Float32Array(nFft)

  // windowSum depends only on (numFrames, nFft, hop), so cache it
  // (accumulation order is identical to the loop, so values match exactly)
  const wsKey = `${numFrames}|${nFft}|${hop}`
  let windowSum = windowSumCache.get(wsKey)
  if (!windowSum) {
    windowSum = new Float32Array(totalLength)
    for (let t = 0; t < numFrames; t++) {
      const start = t * hop
      for (let i = 0; i < nFft; i++) windowSum[start + i] += window[i] * window[i]
    }
    windowSumCache.set(wsKey, windowSum)
  }

  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < numBins; f++) {
      fullR[f] = real[f * numFrames + t]
      fullI[f] = imag[f * numFrames + t]
    }
    for (let f = 1; f < numBins - 1; f++) {
      fullR[nFft - f] = fullR[f]
      fullI[nFft - f] = -fullI[f]
    }
    ifft(outR, outI, fullR, fullI, nFft)
    const start = t * hop
    for (let i = 0; i < nFft; i++) {
      output[start + i] += outR[i] * window[i]
    }
  }

  const outLen = Math.min(length, totalLength - 2 * pad)
  const result = new Float32Array(length)
  for (let i = 0; i < outLen; i++) {
    const j = pad + i
    result[i] = windowSum[j] > 1e-11 ? output[j] / windowSum[j] : 0
  }
  return result
}
