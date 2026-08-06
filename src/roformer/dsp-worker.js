// Run STFT/iSTFT in a Web Worker (for L/R parallelization). Numerically identical to dsp.js.
import { stftTorch, istftTorch } from './dsp.js'

self.onmessage = (e) => {
  const { id, op, a, opts } = e.data
  if (op === 'stft') {
    const { real, imag } = stftTorch(a, opts.nFft, opts.hop)
    // Do the peak scan here as well, in parallel
    let peak = 1e-9
    for (let i = 0; i < real.length; i++) {
      const x = Math.abs(real[i]); if (x > peak) peak = x
      const y = Math.abs(imag[i]); if (y > peak) peak = y
    }
    self.postMessage({ id, real, imag, peak }, [real.buffer, imag.buffer])
  } else {
    // a: one channel of model output (real/imag interleaved); unscale by peak
    const n = a.length / 2
    const real = new Float32Array(n)
    const imag = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      real[i] = a[i * 2] * opts.peak
      imag[i] = a[i * 2 + 1] * opts.peak
    }
    const out = istftTorch(real, imag, opts.frames, opts.nFft, opts.hop, opts.length)
    self.postMessage({ id, out }, [out.buffer])
  }
}
