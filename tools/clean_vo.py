"""Clean a voiceover before editing: rumble high-pass, click repair, de-esser.

usage: python clean_vo.py IN.wav OUT.wav
"""
import sys
import numpy as np, soundfile as sf, scipy.signal as ss

src, dst = sys.argv[1], sys.argv[2]
a, sr = sf.read(src, dtype="float64")
if a.ndim > 1: a = a.mean(1)

# 1) rumble / DC
a = ss.sosfiltfilt(ss.butter(4, 70, "hp", fs=sr, output="sos"), a)

# 2) clicks: sample jumps far above the local level -> replace a short span by cubic interpolation
d = np.abs(np.diff(a)); loc = np.convolve(np.abs(a), np.ones(441) / 441, "same")[1:] + 1e-4
clicks = np.flatnonzero(d / loc > 8)
for c in clicks:
    lo, hi = max(c - 12, 1), min(c + 12, len(a) - 2)
    x = np.r_[lo - 6:lo, hi:hi + 6]
    a[lo:hi] = np.interp(np.arange(lo, hi), x, a[x])
print(f"repaired {len(clicks)} clicks")

# 3) de-esser: STFT gain on 4.5-11 kHz when that band dominates the frame
n = 1024; hopn = 256
f, t, Z = ss.stft(a, fs=sr, nperseg=n, noverlap=n - hopn)
P = np.abs(Z) ** 2
band = (f >= 4500) & (f <= 11000)
ratio = P[band].sum(0) / (P.sum(0) + 1e-12)
level = 10 * np.log10(P.sum(0) + 1e-12)
loud = level > np.percentile(level, 30)
# reduce the band so its share never exceeds 0.22; at most -12 dB
g = np.ones_like(ratio)
over = loud & (ratio > 0.22)
g[over] = np.maximum(10 ** (-12 / 20), np.sqrt(0.22 / ratio[over]))
g = np.convolve(g, np.ones(3) / 3, "same")             # smooth over time (no zipper noise)
Z[band] *= g[None, :]
a = ss.istft(Z, fs=sr, nperseg=n, noverlap=n - hopn)[1][: len(a)]
print(f"de-essed {int(over.sum())} frames ({over.mean() * 100:.1f}%)")

a = a / max(np.abs(a).max(), 1e-9) * 0.95
sf.write(dst, a.astype(np.float32), sr)
