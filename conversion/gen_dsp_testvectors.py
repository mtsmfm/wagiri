"""Generate reference vectors with torch for verifying the JS-side STFT/iSTFT."""
import json
import numpy as np
import torch

SR = 44100
N_FFT = 2048
HOP = 441

rng = np.random.default_rng(42)
n = SR  # 1 second is enough
sig = rng.standard_normal(n).astype(np.float32) * 0.3

x = torch.from_numpy(sig)
window = torch.hann_window(N_FFT)
spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                  window=window, return_complex=True, center=True)
recon = torch.istft(spec, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                    window=window, length=n)

out = {
    'signal': sig.tolist(),
    'numFrames': spec.shape[1],
    'numBins': spec.shape[0],
    # All-bin comparison for the first 5 frames only (to keep JSON small), plus summary stats
    'specRealHead': spec.real[:, :5].numpy().astype(float).tolist(),
    'specImagHead': spec.imag[:, :5].numpy().astype(float).tolist(),
    'specAbsSum': float(spec.abs().sum()),
    'recon': recon.numpy().astype(float).tolist(),
}
with open('dsp_testvectors.json', 'w') as f:
    json.dump(out, f)
print('written', spec.shape)
