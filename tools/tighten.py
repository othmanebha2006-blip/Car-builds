"""Shorten pauses in a voiceover: any quiet run longer than MAXGAP is cut down to KEEP seconds (with short fades)."""
import sys, numpy as np, soundfile as sf
src, dst = sys.argv[1], sys.argv[2]
THR_DB, MAXGAP, KEEP, FADE = -38, 0.20, 0.14, 0.012
a, sr = sf.read(src, dtype="float32")
hop = int(sr * 0.01)
n = len(a) // hop
rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
quiet = 20 * np.log10(rms) < THR_DB
runs, i = [], 0
while i < n:
    if quiet[i]:
        j = i
        while j < n and quiet[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
keep_mask = np.ones(len(a), bool)
for s, e in runs:
    dur = (e - s) * 0.01
    lead = s == 0; tail = e == n
    if lead or tail: target = 0.05
    elif dur > MAXGAP: target = KEEP
    else: continue
    cut_s = s * hop + int(target / 2 * sr) if not lead else 0
    cut_e = e * hop - int(target / 2 * sr) if not tail else len(a)
    if lead: cut_e = e * hop - int(target * sr)
    if tail: cut_s = s * hop + int(target * sr)
    if cut_e > cut_s: keep_mask[cut_s:cut_e] = False
out = a[keep_mask]
# smooth each splice point
idx = np.flatnonzero(np.diff(keep_mask.astype(int)) == 1)  # starts of kept blocks after a cut
pos = np.cumsum(keep_mask)[idx] if len(idx) else []
f = int(FADE * sr)
for p in pos:
    p = int(p)
    if f < p < len(out) - f:
        out[p - f:p] *= np.linspace(1, 0.3, f); out[p:p + f] *= np.linspace(0.3, 1, f)
sf.write(dst, out, sr)
print(f"{len(a)/sr:.2f}s -> {len(out)/sr:.2f}s, {len(runs)} quiet runs")
