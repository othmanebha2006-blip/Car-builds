"""Accurate word timing for captions.

Runs a Zipformer transducer (sherpa-onnx) on the voiceover to get real per-token timestamps, then maps the
script's words onto the recognised words (sequence alignment).  Each word's start is the moment the model
heard it; captions are shown from that moment, so they can never lead the audio.

usage: python align_words.py VO.wav OLD_WORDS.json OUT_WORDS.json MODEL_DIR
"""
import difflib, json, re, sys
import numpy as np, soundfile as sf, sherpa_onnx

wav, old_json, out_json, M = sys.argv[1:5]
a, sr = sf.read(wav, dtype="float32")
if a.ndim > 1: a = a.mean(1)
if sr != 16000:
    import scipy.signal as ss
    a = ss.resample_poly(a, 16000, sr).astype("float32"); sr = 16000

rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{M}/encoder-epoch-99-avg-1.int8.onnx", decoder=f"{M}/decoder-epoch-99-avg-1.int8.onnx",
    joiner=f"{M}/joiner-epoch-99-avg-1.int8.onnx", tokens=f"{M}/tokens.txt", num_threads=4, decoding_method="greedy_search")

# split at pauses (>=0.25 s quiet) so each chunk is short; keep absolute offsets
hop = 160; n = len(a) // hop
db = 20 * np.log10(np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
speech = db > -42
chunks, i = [], 0
while i < n:
    if speech[i]:
        j = i
        while j < n and speech[j:j + 25].any(): j += 1
        chunks.append((max(0, i - 8), min(n, j + 8))); i = j
    else: i += 1

rec_chars = []   # (char, time) for every recognised letter/digit
for s, e in chunks:
    st = rec.create_stream(); st.accept_waveform(sr, a[s * hop:e * hop]); rec.decode_stream(st)
    r = st.result; off = s * hop / sr
    for tok, ts in zip(r.tokens, r.timestamps):
        for ch in re.sub(r"[^a-z0-9]", "", tok.lower()):
            rec_chars.append((ch, off + ts))

from num2words import num2words
def spoken(w):
    """script word -> spoken letters, e.g. '2014-2016' -> 'twentyfourteentwentysixteen', 'N63' -> 'nsixtythree'"""
    w = w.lower()
    def num(m):
        x = m.group(0)
        if "." in x: return " ".join(num2words(int(p)) for p in x.split("."))
        v = int(x)
        if len(x) == 4 and 1900 < v < 2100: return num2words(v, to="year")
        if v in (550, 450, 440) and len(x) == 3 and False: return ""
        return num2words(v)
    w = re.sub(r"\d+(?:\.\d+)?", num, w)
    return re.sub(r"[^a-z]", "", w)

script = json.load(open(old_json))
sc_chars = []   # (char, word index)
for k, w in enumerate(script):
    for ch in spoken(w["w"]):
        sc_chars.append((ch, k))
A = "".join(c for c, _ in sc_chars); B = "".join(c for c, _ in rec_chars)
B = re.sub(r"\d", "", B) if False else B
sm = difflib.SequenceMatcher(a=A, b=B, autojunk=False)
first_time = {}
matched_words = set()
for i1, j1, size in sm.get_matching_blocks():
    for q in range(size):
        k = sc_chars[i1 + q][1]
        t = rec_chars[j1 + q][1]
        if k not in first_time or (i1 + q) == min(ii for ii, (_, kk) in enumerate(sc_chars) if kk == k):
            pass
        # the word's start = time of its first matched character
        if k not in first_time: first_time[k] = (i1 + q, t)
starts = [None] * len(script)
first_char_idx = {}
for ii, (_, k) in enumerate(sc_chars): first_char_idx.setdefault(k, ii)
for k, (ci, t) in first_time.items():
    # accept only if the matched char is within the first 3 letters of the word (word onset really heard)
    if ci - first_char_idx[k] <= 2: starts[k] = t; matched_words.add(k)
matched = len(matched_words)
rec_words = rec_chars
# fill gaps (words not matched at their onset) between known neighbours
for k in range(len(starts)):
    if starts[k] is None:
        p = next(((q, starts[q]) for q in range(k - 1, -1, -1) if starts[q] is not None), (-1, 0.0))
        nx = next(((q, starts[q]) for q in range(k + 1, len(starts)) if starts[q] is not None), None)
        if nx: starts[k] = p[1] + (nx[1] - p[1]) * (k - p[0]) / (nx[0] - p[0])
        else: starts[k] = p[1] + 0.3
# never before the voice: if the audio is still silent at the timestamp, move to the frame where the voice begins
hop5 = sr // 200; n5 = len(a) // hop5
db5 = 20 * np.log10(np.sqrt(np.mean(a[: n5 * hop5].reshape(n5, hop5) ** 2, 1)) + 1e-9)
QUIET = -40
for k in range(len(starts)):
    f = int(starts[k] * 200)
    # a silence that begins at (or within 60 ms after) the timestamp means the word is not audible yet:
    # it starts where that silence ends (previous word's tail, a pause, or a 'p/t/k' closure)
    for g in range(f, min(f + 13, n5)):
        if db5[g] < QUIET:
            h = g
            while h < n5 - 1 and db5[h] < QUIET: h += 1
            if h - g >= 6 or g == f:          # >= 30 ms of silence (or already silent at the timestamp)
                f = h
            break
    starts[k] = max(starts[k], f / 200)
SAFETY = 0.02   # audio always leads the caption
starts = [t + SAFETY for t in starts]
for k in range(len(starts)):          # strict: never start inside a quiet frame (e.g. a 10 ms gap before a 'b')
    f = int(round(starts[k] * 200))
    while f < n5 - 2 and db5[f:f + 2].max() < QUIET: f += 1
    starts[k] = max(starts[k], f / 200)
# monotonic
for k in range(1, len(starts)):
    starts[k] = max(starts[k], starts[k - 1] + 0.04)

# word end: next word start, or when the voice goes quiet after this word
ends = []
for k, st in enumerate(starts):
    nxt = starts[k + 1] if k + 1 < len(starts) else len(a) / sr
    fi = int(st * 100); end = nxt
    q = 0
    for f in range(fi + 12, int(nxt * 100)):
        q = q + 1 if (f < len(db) and db[f] < -42) else 0
        if q >= 8: end = (f - 7) / 100; break
    ends.append(end)

out = [{"w": w["w"], "s": round(starts[k], 3), "e": round(ends[k], 3)} for k, w in enumerate(script)]
json.dump(out, open(out_json, "w"), indent=0)
print(f"{len(script)} script words, {len(rec_chars)} recognised letters, {matched} words matched at their onset")
shift = [round(starts[k] - script[k]["s"], 2) for k in range(len(script))]
print("old estimate - new (s): mean %+.2f  max early %+.2f" % (np.mean(shift), max(shift)))
