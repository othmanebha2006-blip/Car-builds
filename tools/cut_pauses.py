"""Remove only clear pauses from a voiceover; nothing else is changed.

A pause = a run of true silence (every 5 ms frame below THR dB) lasting at least MIN_PAUSE s.
Each pause is shortened to KEEP s (half kept after the last word, half before the next), joined with a short
crossfade inside the silence.  Gaps shorter than MIN_PAUSE (the natural spaces of smooth reading) are not touched,
and the voice itself is not processed (no filtering, no de-essing, no gain change).

usage: python cut_pauses.py IN.wav OUT.wav [THR_DB=-47] [MIN_PAUSE=0.15] [KEEP=0.10]
"""
import sys
import numpy as np, soundfile as sf
src, dst = sys.argv[1], sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else -47
MIN_PAUSE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
KEEP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.10
a, sr = sf.read(src, dtype="float32", always_2d=True)
info = sf.info(src)
m = a.mean(1); h = int(0.005 * sr); n = len(m) // h
db = 20 * np.log10(np.sqrt((m[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
q = db < THR
cuts = []; i = 0
while i < n:
    if q[i]:
        j = i
        while j < n and q[j]: j += 1
        if (j - i) * h / sr >= MIN_PAUSE and i > 0 and j < n:          # never the leading/trailing silence
            half = int(KEEP / 2 * sr)
            cuts.append((i * h + half, j * h - half))
        i = j
    else: i += 1
xf = int(0.008 * sr); out = []; pos = 0
for s, e in cuts:
    out.append(a[pos:s]); pos = e
out.append(a[pos:])
y = out[0]
for seg in out[1:]:                                                    # crossfade inside the silence
    r = np.linspace(0, 1, xf)[:, None]
    y = np.concatenate([y[:-xf], y[-xf:] * (1 - r) + seg[:xf] * r, seg[xf:]])
sf.write(dst, y, sr, subtype=info.subtype if info.subtype in ("PCM_16", "PCM_24", "FLOAT") else "PCM_16")
removed = (len(a) - len(y)) / sr
print(f"{len(a)/sr:.2f}s -> {len(y)/sr:.2f}s: {len(cuts)} pauses shortened to {KEEP*1000:.0f} ms, {removed:.2f}s removed")
for s, e in cuts: print(f"  pause at {s/sr - KEEP/2:.2f}s: {(e - s)/sr + KEEP:.2f}s -> {KEEP:.2f}s")
