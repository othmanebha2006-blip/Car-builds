"""Check caption start times against the voice audio: no caption may start while the voice is still silent,
and every word that follows a pause must start at/after the physical voice onset."""
import json, sys, numpy as np, soundfile as sf
words = json.load(open(sys.argv[1])); a, sr = sf.read(sys.argv[2])
if a.ndim > 1: a = a.mean(1)
if sr != 16000:   # exact 5 ms frames (80 samples at 16 kHz) so frame index == time * 200 with no drift
    import scipy.signal as ss
    a = ss.resample_poly(a, 16000, sr); sr = 16000
off = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0          # audio offset of the voice in this file
hop = sr // 200; n = len(a) // hop
db = 20 * np.log10(np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
early, post = [], []
for w in words:
    t = w["S"] if "S" in w else w["s"] + off
    f = int(round(t * 200))
    if f >= len(db) - 2: continue
    if db[f:f + 2].max() < -40: early.append((w["w"], round(t, 2), "starts while the voice is silent"))
    # a pause (>=100 ms quiet) that is still going on at the caption start = caption shown before the word
    g = f; 
    while g > 0 and db[g - 1] < -40: g -= 1
    h = f
    while h < len(db) - 1 and db[h] < -40: h += 1
    if db[f] < -40 and h - g >= 20: early.append((w["w"], round(t, 2), f"shown {(h - f) / 200:.3f}s before the voice resumes"))
    # words that follow a pause: measure caption start vs the end of that pause
    lo = f
    while lo > 0 and db[lo - 1] >= -40 and f - lo < 12: lo -= 1
    q = lo
    while q > 0 and db[q - 1] < -40: q -= 1
    if lo - q >= 20 and f - lo < 12: post.append((t - lo / 200))
post = np.array(post)
print(f"{len(words)} captions; {len(post)} start right after a silence/closure: caption minus voice onset "
      f"min {post.min():+.3f}s, mean {post.mean():+.3f}s, max {post.max():+.3f}s")
print("captions shown BEFORE the voice:", early if early else "none")
