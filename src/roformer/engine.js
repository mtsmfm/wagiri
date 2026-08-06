// Vocals/BGM separation engine based on windowed-roformer (our own ONNX conversion).
// Model I/O: stft_in / masked_stft (1, 2, 1025, 801, 2)
// STFT/iSTFT run in parallel in dsp-worker.js (2 workers, one per L/R channel).

export const ROFORMER = {
  SAMPLE_RATE: 44100,
  N_FFT: 2048,
  HOP: 441,
  CHUNK: 8 * 44100,
  T_FRAMES: 801,
  N_BINS: 1025,
}

export class RoformerEngine {
  constructor({ ort, onProgress = () => {}, executionProviders = ['webgpu', 'wasm'], sessionOptions = {} }) {
    this.ort = ort
    this.onProgress = onProgress
    this.executionProviders = executionProviders
    this.sessionOptions = sessionOptions
    this.session = null
  }

  async loadModel(modelBuffer) {
    this.session = await this.ort.InferenceSession.create(modelBuffer, {
      executionProviders: this.executionProviders,
      graphOptimizationLevel: 'all',
      ...this.sessionOptions,
    })
  }

  /**
   * @param {Float32Array} left 44100Hz
   * @param {Float32Array} right 44100Hz
   * @returns {{vocals: {left, right}, bgm: {left, right}}}
   */
  async separate(left, right) {
    const { CHUNK, N_FFT, HOP, T_FRAMES, N_BINS } = ROFORMER
    const n = left.length
    const numChunks = Math.ceil(n / CHUNK)
    const vocalsL = new Float32Array(n)
    const vocalsR = new Float32Array(n)

    const tStart = performance.now()
    const timings = { stftMs: 0, runMs: 0, istftMs: 0, perChunkRunMs: [] }

    // Chunk pipelining: while the GPU runs inference on chunk ci, the workers compute
    // the STFT of ci+1 and the iSTFT of ci-1. Chunks are numerically independent,
    // so the result is exactly identical to the sequential version.
    const stftPair = (ci) => {
      const start = ci * CHUNK
      const chunkL = new Float32Array(CHUNK)
      const chunkR = new Float32Array(CHUNK)
      chunkL.set(left.subarray(start, Math.min(start + CHUNK, n)))
      chunkR.set(right.subarray(start, Math.min(start + CHUNK, n)))
      return Promise.all([
        this.dspCall(0, 'stft', chunkL, { nFft: N_FFT, hop: HOP }),
        this.dspCall(1, 'stft', chunkR, { nFft: N_FFT, hop: HOP }),
      ])
    }
    const istftJobs = []

    let specP = stftPair(0)
    for (let ci = 0; ci < numChunks; ci++) {
      const start = ci * CHUNK
      const tStft = performance.now()
      const [specL, specR] = await specP
      timings.stftMs += performance.now() - tStft

      // The model is input-scale invariant (band split starts with RMSNorm), so
      // peak-normalize the input to avoid fp16 overflow and undo it on the output
      const peak = Math.max(specL.peak, specR.peak)
      const scale = 1 / peak

      // (1, 2, N_BINS, T_FRAMES, 2) contiguous
      const input = new Float32Array(2 * N_BINS * T_FRAMES * 2)
      packChannel(input, 0, specL, scale)
      packChannel(input, 1, specR, scale)

      // With enableGraphCapture, the same GPU buffers must be reused for both input
      // and output (ORT requires external buffers). Otherwise use normal CPU tensors.
      const dims = [1, 2, N_BINS, T_FRAMES, 2]
      let tensor, fetches
      if (this.sessionOptions.enableGraphCapture) {
        const device = this.ort.env.webgpu.device
        const USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        if (!this.inputGpuBuffer) {
          this.inputGpuBuffer = device.createBuffer({ size: input.byteLength, usage: USAGE })
          this.outputGpuBuffer = device.createBuffer({ size: input.byteLength, usage: USAGE })
        }
        device.queue.writeBuffer(this.inputGpuBuffer, 0, input.buffer, 0, input.byteLength)
        tensor = this.ort.Tensor.fromGpuBuffer(this.inputGpuBuffer, { dataType: 'float32', dims })
        fetches = { masked_stft: this.ort.Tensor.fromGpuBuffer(this.outputGpuBuffer, { dataType: 'float32', dims }) }
      } else {
        tensor = new this.ort.Tensor('float32', input, dims)
      }
      // Kick off the next chunk's STFT before the GPU run (workers overlap with GPU inference)
      if (ci + 1 < numChunks) specP = stftPair(ci + 1)
      const tRun = performance.now()
      const out = await this.session.run({ stft_in: tensor }, fetches)
      let masked
      if (this.sessionOptions.enableGraphCapture) {
        masked = await this.readbackOutput(input.byteLength)
      } else {
        masked = await out.masked_stft.getData()
      }
      const runMs = performance.now() - tRun
      timings.runMs += runMs
      timings.perChunkRunMs.push(Math.round(runMs))

      // Queue this chunk's iSTFT as a job without awaiting (overlaps with the next GPU run)
      const half = N_BINS * T_FRAMES * 2
      const iopts = { peak, frames: T_FRAMES, nFft: N_FFT, hop: HOP, length: CHUNK }
      istftJobs.push(
        Promise.all([
          this.dspCall(0, 'istft', masked.slice(0, half), iopts),
          this.dspCall(1, 'istft', masked.slice(half, 2 * half), iopts),
        ]).then(([outL, outR]) => {
          vocalsL.set(outL.out.subarray(0, Math.min(CHUNK, n - start)), start)
          vocalsR.set(outR.out.subarray(0, Math.min(CHUNK, n - start)), start)
        })
      )

      this.onProgress({ progress: (ci + 1) / numChunks, currentSegment: ci + 1, totalSegments: numChunks })
    }
    const tIstft = performance.now()
    await Promise.all(istftJobs)
    timings.istftMs += performance.now() - tIstft

    const bgmL = new Float32Array(n)
    const bgmR = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      bgmL[i] = left[i] - vocalsL[i]
      bgmR[i] = right[i] - vocalsR[i]
    }

    // Output sanity: detect NaN / all-zero output and pass it to the caller
    let nan = 0, sumV = 0, sumM = 0
    for (let i = 0; i < n; i++) {
      const v = vocalsL[i]
      if (Number.isNaN(v)) nan++
      sumV += v * v
      sumM += left[i] * left[i]
    }
    for (const k of ['stftMs', 'runMs', 'istftMs']) timings[k] = Math.round(timings[k])
    timings.totalMs = Math.round(performance.now() - tStart)
    const sanity = {
      nanCount: nan,
      vocalsRms: Math.sqrt(sumV / n),
      mixRms: Math.sqrt(sumM / n),
      timings,
    }
    console.log('[wagiri] separation sanity:', sanity)

    return {
      vocals: { left: vocalsL, right: vocalsR },
      bgm: { left: bgmL, right: bgmR },
      sanity,
    }
  }

  // DSP worker pool (2 workers, for L/R). The buffer of `a` is transferred
  dspCall(slot, op, a, opts) {
    if (!this.dspWorkers) {
      this.dspWorkers = [0, 1].map(() => new Worker(new URL('./dsp-worker.js', import.meta.url), { type: 'module' }))
      this.dspPending = new Map()
      this.dspId = 0
      for (const w of this.dspWorkers) {
        w.onmessage = (e) => {
          const cb = this.dspPending.get(e.data.id)
          this.dspPending.delete(e.data.id)
          cb(e.data)
        }
      }
    }
    return new Promise((resolve) => {
      const id = this.dspId++
      this.dspPending.set(id, resolve)
      this.dspWorkers[slot].postMessage({ id, op, a, opts }, [a.buffer])
    })
  }

  // Read outputGpuBuffer back to the CPU via a staging buffer
  async readbackOutput(byteLength) {
    const device = this.ort.env.webgpu.device
    if (!this.stagingBuffer) {
      this.stagingBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    }
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(this.outputGpuBuffer, 0, this.stagingBuffer, 0, byteLength)
    device.queue.submit([enc.finish()])
    await this.stagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(this.stagingBuffer.getMappedRange().slice(0))
    this.stagingBuffer.unmap()
    return data
  }
}

function packChannel(input, ch, spec, scale) {
  const { N_BINS, T_FRAMES } = ROFORMER
  const base = ch * N_BINS * T_FRAMES * 2
  const { real, imag } = spec
  for (let i = 0; i < N_BINS * T_FRAMES; i++) {
    input[base + i * 2] = real[i] * scale
    input[base + i * 2 + 1] = imag[i] * scale
  }
}

