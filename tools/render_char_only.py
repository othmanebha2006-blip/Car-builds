"""Character-only video: the rigged Mecmation character on a plain white background, lip-synced to a voiceover,
with exactly the motion used in the Silverado / G70 videos (tools/render_silverado.py).

usage: python render_char_only.py PROJECT_DIR [OUT.mp4]
needs PROJECT_DIR/build/vo_tight.wav, build/words.json (align_words.py) and gestures.json"""
import json, math, os, random, re, subprocess, sys
from multiprocessing import Pool
import numpy as np, soundfile as sf
from PIL import Image
from num2words import num2words

ROOT = "/home/user/Car-builds"
P = os.path.abspath(sys.argv[1]); BUILD = f"{P}/build"
W, H, FPS = 1080, 1920, 30
CHAR_H = 1480                      # character height on screen (feet to hat)
FEET = 1830
VO_DUR = sf.info(f"{BUILD}/vo_tight.wav").duration
TOTAL = VO_DUR + 0.8

sys.path.insert(0, f"{ROOT}/tools")
import stick_rig
from stick_rig import StickRig, POSES
stick_rig.MOUTH_SCALE = 1.0
RIG = StickRig(f"{ROOT}/projects/rs5/cut/character.png")

# ---------------------------------------------------------------- words (onset snapped to audible voice)
WORDS = json.load(open(f"{BUILD}/words.json"))
_v, _vsr = sf.read(f"{BUILD}/vo_tight.wav", dtype="float32")
import scipy.signal as _ss
_v = _ss.resample_poly(_v, 16000, _vsr).astype("float32")
_n5 = len(_v) // 80
_db5 = 20 * np.log10(np.sqrt(np.mean(_v[: _n5 * 80].reshape(_n5, 80) ** 2, 1)) + 1e-9)
for w in WORDS:
    f = int(round(w["s"] * 200))
    while f < _n5 - 2 and _db5[f:f + 2].max() < -40: f += 1
    w["S"] = max(w["s"], f / 200); w["E"] = max(w["e"], w["S"] + 0.05)

def at(token, n=1):
    k = 0; tok = token.lower().strip(".,?!")
    for w in WORDS:
        ww = w["w"].lower().strip(".,?!")
        if ww == tok or (any(ch.isdigit() for ch in tok) and ww.startswith(tok)):
            k += 1
            if k == n: return w["S"]
    raise KeyError(token)

# ---------------------------------------------------------------- mouth (same viseme track as the car videos)
def _spoken(w):
    w = w.lower()
    def num(m):
        x = m.group(0).replace(",", "")
        v = int(x); return num2words(v, to="year") if len(x) == 4 and 1900 < v < 2100 else num2words(v)
    return re.sub(r"[^a-z]", "", re.sub(r"\d[\d,]*", num, w))

def _visemes(word):
    s = _spoken(word); out = []; i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in ("oo", "ou", "ow", "qu"): out.append(4); i += 2; continue
        if two in ("ee", "ea", "ch", "sh"): out.append(5); i += 2; continue
        if two in ("ph",): out.append(6); i += 2; continue
        if two in ("th",): out.append(2); i += 2; continue
        c = s[i]; i += 1
        out.append(1 if c in "mbp" else 6 if c in "fv" else 4 if c in "wu" else 7 if c == "o" else 3 if c in "ai"
                   else 5 if c == "j" else 2)
    keep = [v for k, v in enumerate(out) if k == 0 or v != out[k - 1]]
    return keep or [2]

NF = int(TOTAL * FPS) + 2
MOUTH = np.ones(NF, int)
for w in WORDS:
    f0, f1 = int(round(w["S"] * FPS)), max(int(round(w["E"] * FPS)), int(round(w["S"] * FPS)) + 2)
    seq = _visemes(w["w"]); n = max(1, min(len(seq), (f1 - f0) // 2))
    pick = [seq[int(k * len(seq) / n)] for k in range(n)]
    for k, v in enumerate(pick):
        MOUTH[f0 + (f1 - f0) * k // n: f0 + (f1 - f0) * (k + 1) // n] = v

# the mouth rests whenever the voice is really silent during the frame (no open mouth over pauses)
for _fi in range(NF):
    q0, q1 = int(_fi / FPS * 200), int((_fi + 1) / FPS * 200)
    if q0 >= _n5 or _db5[q0:min(q1, _n5)].max() < -45: MOUTH[_fi] = 1

# ---------------------------------------------------------------- body motion (Silverado / G70 rules)
def ease_io(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def _back(x, s=1.25): x = min(max(x, 0.0), 1.0); x -= 1; return x * x * ((s + 1) * x + s) + 1
DEFAULT, MOVE, HOLD = "tight", 0.35, 1.2
SPEC = json.load(open(f"{P}/gestures.json"))
GEST = sorted([(at(g[0], g[1]) - 0.12, g[2], g[3] if len(g) > 3 else HOLD) for g in SPEC["gestures"]])
BROWS = [at(w, n) for w, n in SPEC.get("brows", [])]
NODS = [at(w, n) for w, n in SPEC.get("nods", [])]
def _pv(name): p = POSES[name]; return np.array([*p["L"], *p["R"]], float)

def arm_pose(t):
    cur = _pv(DEFAULT)
    for k, (st, name, hold) in enumerate(GEST):
        if st > t: break
        nxt = GEST[k + 1][0] if k + 1 < len(GEST) else 1e9
        start = cur.copy(); cur = start + (_pv(name) - start) * _back((t - st) / MOVE)
        end_hold = min(st + MOVE + hold, nxt)
        if t > end_hold and nxt - end_hold > 0.15:
            cur = cur + (_pv(DEFAULT) - cur) * ease_io((t - end_hold) / MOVE)
    return {"L": (cur[0], cur[1]), "R": (cur[2], cur[3])}

def _side(t):
    s = 0.0
    for st, name, hold in GEST:
        if st <= t <= st + MOVE + hold: s = -1.0 if name.endswith("_L") else 1.0 if name.endswith("_R") else 0.0
    return s

_rng = random.Random(11); BLINKS = []; _t = 1.4
while _t < TOTAL: BLINKS.append(_t); _t += _rng.uniform(2.8, 4.2)

def head(t):
    dx = 13 * math.sin(2 * math.pi * t / 5.7) + 5 * math.sin(2 * math.pi * t / 2.3 + 1.3)
    tilt = max(-3.0, min(3.0, 2.5 * _side(t) + 0.6 * math.sin(2 * math.pi * t / 4.1)))
    dy = 0.0
    for nt in NODS:
        x = (t - nt) / 0.4
        if 0 <= x <= 1: dy = 7 * math.sin(math.pi * x)
    brow = 0.0
    for bt in BROWS:
        x = (t - bt + 0.05) / 0.55
        if 0 <= x <= 1: brow = max(brow, 11 * math.sin(math.pi * min(1, x * 1.4)) if x < 0.7 else 11 * (1 - ease_io((x - 0.7) / 0.3)))
    return dx, dy, tilt, brow, any(0 <= t - b < 0.13 for b in BLINKS)

SCALE = CHAR_H / RIG.h
def render(fi):
    t = fi / FPS
    dx, dy, tilt, brow, blink = head(t)
    im = RIG.pose(arm_pose(t), int(MOUTH[min(fi, NF - 1)]), blink, brow, head_dx=dx, head_dy=dy, tilt=tilt, sway=0.25 * dx)
    w, h = int(im.width * SCALE), int(im.height * SCALE)
    fr = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    fr.alpha_composite(im.resize((w, h), Image.LANCZOS), (int(W / 2 - w / 2), int(FEET - h)))
    return fr.convert("RGB").tobytes()

if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else f"{P}/character_video.mp4"
    if os.environ.get("PREVIEW"):
        for s in os.environ["PREVIEW"].split(","):
            Image.frombytes("RGB", (W, H), render(int(float(s) * FPS))).save(f"{BUILD}/prev_{s}.png")
        sys.exit()
    af = "apad" if SPEC.get("keep_voice") else "loudnorm=I=-14:TP=-1:LRA=11,apad"   # keep_voice: the voice exactly as delivered
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{BUILD}/vo_tight.wav", "-af", af,
                    "-ar", "44100", "-t", f"{TOTAL:.3f}", f"{BUILD}/vo_norm.wav"], check=True)
    nf = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", f"{BUILD}/vo_norm.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for k, b in enumerate(pool.imap(render, range(nf), chunksize=8)):
            ff.stdin.write(b)
            if k % 300 == 0: print(f"frame {k}/{nf}", flush=True)
    ff.stdin.close(); ff.wait(); print("done", out, f"{TOTAL:.1f}s")
