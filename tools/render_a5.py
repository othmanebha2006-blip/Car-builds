"""Render the Audi A5 2.0T short (1080x1920, 30 fps): cars on top, rigged stick-arm character below, captions, shine, SFX."""
import json, math, os, random, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/user/Car-builds"
P = f"{ROOT}/projects/a5"
CUT, BUILD = f"{P}/cut", f"{P}/build"
FONT = f"{ROOT}/tools/fonts/Montserrat-900.ttf"
W, H, FPS = 1080, 1920, 30
BLUE, GREEN, BLACK, WHITE = (30, 144, 255), (34, 197, 94), (12, 12, 12), (255, 255, 255)

# ---------------------------------------------------------------- timeline
_W0 = json.load(open(f"{BUILD}/words.json"))
_mk = next(k for k, w in enumerate(_W0) if w["w"] == "market.")
HOOK_END_VO = (_W0[_mk]["e"] + _W0[_mk + 1]["s"]) / 2       # cut between "This is the Ford Flex EcoBoost." and "At first glance"
USE_ENGINE = True                                             # user: clip sound after the hook ("used market.")
ENG_DUR = (sf.info(f"{P}/raw/engine_sound.wav").duration - 0.1) if USE_ENGINE else 0.0
ENG_START = HOOK_END_VO + 0.05 if USE_ENGINE else 1e9
SHIFT = (ENG_DUR + 0.2) if USE_ENGINE else 0.0                                         # voice resumes right after the engine sound
VO_DUR = sf.info(f"{BUILD}/vo_tight.wav").duration
TOTAL = VO_DUR + SHIFT + 1.4

def vt(t):  # voiceover time -> final time
    return t if t < HOOK_END_VO else t + SHIFT

WORDS = json.load(open(f"{BUILD}/words.json"))
_v, _vsr = sf.read(f"{BUILD}/vo_tight.wav", dtype="float32")
import scipy.signal as _ss
_v = _ss.resample_poly(_v, 16000, _vsr).astype("float32"); _vsr = 16000   # exact 5 ms frames (80 samples)
_h = 80; _n5 = len(_v) // _h
_db5 = 20 * np.log10(np.sqrt(np.mean(_v[: _n5 * _h].reshape(_n5, _h) ** 2, 1)) + 1e-9)
for w in WORDS:
    # caption = the word's measured onset; only if the voice is still silent there, move to where it becomes audible
    f = int(round(w["s"] * 200))
    while f < _n5 - 2 and _db5[f:f + 2].max() < -40: f += 1
    w["s"] = max(w["s"], f / 200)
    for _ in range(6):                      # the video frame that shows the caption must already have voice
        fr_t = math.ceil(vt(w["s"]) * FPS - 1e-9) / FPS - (vt(w["s"]) - w["s"])
        q = int(round(fr_t * 200))
        if q >= _n5 - 3 or _db5[max(q - 1, 0):q + 3].min() >= -40: break      # whole window -5..+15 ms voiced
        while q < _n5 - 3 and _db5[max(q - 1, 0):q + 3].min() < -40: q += 1
        w["s"] = q / 200
    w["S"], w["E"] = vt(w["s"]), vt(max(w["e"], w["s"] + 0.05))

def at(token, n=1, end=False):
    """final time of the n-th word equal to token (prefix match when token has a digit)"""
    k = 0
    tok = token.lower().strip(".,?!")
    for w in WORDS:
        ww = w["w"].lower().strip(".,?!")
        if ww == tok or (any(ch.isdigit() for ch in tok) and ww.startswith(tok)):
            k += 1
            if k == n:
                return w["E"] if end else w["S"]
    raise KeyError(token)

# ---------------------------------------------------------------- assets
MISSING = []
def load(name):
    if os.path.exists(f"{CUT}/{name}.png"):
        return Image.open(f"{CUT}/{name}.png").convert("RGBA")
    MISSING.append(name)
    ph = Image.new("RGBA", (900, 520) if not name.startswith(("part", "logo")) else (500, 500), (0, 0, 0, 0))
    d = ImageDraw.Draw(ph); d.rounded_rectangle((4, 4, ph.width - 4, ph.height - 4), 40, fill=(120, 120, 128, 255), outline=(40, 40, 40, 255), width=8)
    d.text((30, ph.height // 2 - 20), name, fill=(255, 255, 255, 255), font=ImageFont.truetype(FONT, 40))
    return ph

IMG = {n: load(n) for n in [
    "a5_01_front34", "a5_02_rear34", "a5_03_side", "a5_04_front", "a5_05_rear", "a5_06_hoodopen",
    "a5_07_engine", "a5_08_front34_low", "part_intake", "part_intercooler", "part_downpipe", "part_tune", "part_k04",
    "logo_audi"]}
BG = Image.open(f"{P}/raw/background.png").convert("RGB").resize((W, H))

# character variants: mouth 0..3 x blink
sys.path.insert(0, f"{ROOT}/tools")
from toon_rig import ToonRig, POSES
RIG = ToonRig(f"{CUT}/character_v2.png")

# voice energy for lip-sync (in final time)
vo, VSR = sf.read(f"{BUILD}/vo_tight.wav", dtype="float32")
hop = VSR // 100
_n = len(vo) // hop
ENERGY = np.sqrt(np.mean(vo[: _n * hop].reshape(_n, hop) ** 2, 1))
ENERGY = ENERGY / np.percentile(ENERGY, 95)

def mouth_level(t):
    if ENG_START - 0.05 < t < ENG_START + ENG_DUR + 0.1:
        return 0
    v = t if t < ENG_START else t - SHIFT
    i = int(v * 100)
    if i < 0 or i >= len(ENERGY):
        return 0
    e = ENERGY[max(0, i - 1): i + 2].mean()
    return 0 if e < 0.12 else 1 if e < 0.35 else 2 if e < 0.7 else 3

# ---------------------------------------------------------------- helpers
def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_io(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def back(x):
    x = min(max(x, 0), 1); c1 = 1.9; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

def pop(t, t0, dur=0.24, start=0.55):
    """scale for a pop-in that starts at t0"""
    return start + (1 - start) * back((t - t0) / dur)

def fit(im, bw, bh):
    s = min(bw / im.width, bh / im.height)
    return s

@lru_cache(maxsize=4096)
def scaled(name, w, h):
    return IMG[name].resize((max(1, w), max(1, h)), Image.LANCZOS)

# white flicker measured from the reference (30 fps): fade to white, white silhouette, back, white again
SHINE_PATTERN = [0.33] + [0.33, 0.67, 1.0, 0.67, 0.33, 0.0] * 3   # measured frame-by-frame from the reference

def shine_level(t, times):
    for st in times:
        k = int(round((t - st) * FPS))
        if 0 <= k < len(SHINE_PATTERN):
            return SHINE_PATTERN[k]
    return 0.0

def place(frame, name, cx, cy, bw, bh, scale=1.0, alpha=1.0, flash=0.0):
    im = IMG[name]; s = fit(im, bw, bh) * scale
    w, h = int(im.width * s), int(im.height * s)
    if w < 2 or h < 2:
        return
    # quantise sizes to keep the cache effective
    w -= w % 2; h -= h % 2
    sm = scaled(name, w, h)
    if alpha < 1:
        sm = sm.copy(); sm.putalpha(sm.getchannel("A").point(lambda a: int(a * alpha)))
    if flash > 0:
        a_ch = sm.getchannel("A")
        sm = Image.blend(sm.convert("RGB"), Image.new("RGB", sm.size, (255, 255, 255)), flash).convert("RGBA")
        sm.putalpha(a_ch)
    frame.alpha_composite(sm, (int(cx - w / 2), int(cy - h / 2)))

@lru_cache(maxsize=2048)
def font(size):
    return ImageFont.truetype(FONT, size)

@lru_cache(maxsize=4096)
def text_img(text, size, fill, stroke=0, stroke_fill=BLACK, pill=False):
    f = font(size)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    pad = int(size * 0.45) if pill else 0
    im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if pill:
        d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=int(size * 0.5), fill=(10, 10, 10, 255))
    d.text((pad - l, pad - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im

def put_text(frame, text, cx, cy, size, fill, stroke=0, scale=1.0, pill=False, anchor="c"):
    if not text:
        return
    im = text_img(text, size, fill, stroke, BLACK, pill)
    if abs(scale - 1) > 0.01:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    x = cx - im.width / 2 if anchor == "c" else cx
    frame.alpha_composite(im, (int(x), int(cy - im.height / 2)))
    return im.width

def typed(text, t, t0, cps=20):
    n = int(max(0, t - t0) * cps)
    return text[:n] + ("|" if n < len(text) and int(t * 6) % 2 == 0 else "")

# ---------------------------------------------------------------- scene plan (top half)
CAR_Y = 500
RED = (226, 38, 38)
W_ = lambda w, n=1: at(w, n)
T = dict(
    hidden=W_("hidden"), glance=W_("glance,"), talking=W_("talking"), years=W_("2008"), because=W_("Because"),
    they=W_("They"), but=W_("But"), liter=W_("EA888"), hp208=W_("208"), lb258=W_("258"), potential=W_("potential"),
    mod=W_("mod"), start=W_("Start"), intake=W_("intake,"), ic=W_("intercooler,"), dp=W_("downpipe,"), stage=W_("Stage"),
    thats=W_("That's"), around=W_("approach"), hp_end=at("horsepower", 2, end=True), k04=W_("K04"),
    sudden=W_("Suddenly,"), becomes=W_("becomes"), basically=W_("Basically,"), would=W_("Would"),
)
SCENES = [  # (start, kind, args) -- one shot held ~5-6 s, as in the reference videos
    (0.0, "logo_car", "a5_01_front34"),
    (T["hidden"], "car", "a5_08_front34_low"),
] + ([(ENG_START, "engine", None)] if USE_ENGINE else []) + [
    (T["glance"], "car", "a5_03_side"),
    (T["talking"], "year", "a5_01_front34"),
    (T["because"], "car", "a5_05_rear"),
    (T["they"], "car", "a5_04_front"),
    (T["but"], "car", "a5_08_front34_low"),
    (T["liter"], "spec", None),
    (T["potential"], "car", "a5_02_rear34"),
    (T["start"], "mods", None),
    (T["thats"], "hp", "a5_08_front34_low"),
    (T["k04"], "feature", ("part_k04", "K04 TURBO", "a5_02_rear34")),
    (T["sudden"], "car", "a5_03_side"),
    (T["becomes"], "car", "a5_08_front34_low"),
    (T["basically"], "logo_car", "a5_02_rear34"),
    (T["would"], "logo_car", "a5_01_front34"),
]
SCENE_T = [s[0] for s in SCENES]
SHINES = [T["hidden"], T["years"] + 0.9, T["would"] + 0.35]
MODS = [(T["intake"], "part_intake", "INTAKE"), (T["ic"], "part_intercooler", "INTERCOOLER"),
        (T["dp"], "part_downpipe", "DOWNPIPE"), (T["stage"], "part_tune", "STAGE 2 TUNE")]
MOD_MODE = (T["start"] - 0.15, T["thats"] - 0.1)
FLASHES = [ENG_START] if USE_ENGINE else []
ENG_SHOTS = ["a5_05_rear", "a5_02_rear34"]
LOGO = "logo_audi"
TITLE = ("2008-2016", "AUDI A5 2.0T")

def shake(t, amp, seed=0):
    r = random.Random(int(t * FPS) * 7 + seed)
    return r.uniform(-amp, amp), r.uniform(-amp, amp)

def draw_scene(fr, t):
    i = max(k for k, s in enumerate(SCENE_T) if s <= t)
    t0, kind, arg = SCENES[i]
    nxt = SCENE_T[i + 1] if i + 1 < len(SCENE_T) else TOTAL
    dt, life = t - t0, max(nxt - t0, 0.1)
    s = pop(t, t0) * (1 + 0.035 * min(dt / life, 1))   # pop-in + slow push-in
    fl = shine_level(t, SHINES)
    if kind == "car":
        place(fr, arg, 540, CAR_Y, 960, 520, s, flash=fl)
    elif kind == "logo_car":
        place(fr, LOGO, 540, 215, 420, 150, pop(t, t0 + 0.05))
        place(fr, arg, 540, 590, 900, 420, s, flash=fl)
    elif kind == "diag":
        place(fr, arg[0], 330, 300, 560, 300, pop(t, t0))
        place(fr, arg[1], 660, 640, 760, 400, pop(t, t0 + 0.12) * (1 + 0.03 * dt / life))
    elif kind == "card":
        lines, col, car = arg
        y = 170 if car else 330
        for j, (txt, size) in enumerate(lines):
            put_text(fr, txt, 540, y, size, col, 7, pop(t, t0 + 0.06 * j))
            y += int(size * 1.05)
        if car:
            place(fr, car, 540, 640, 900, 400, pop(t, t0 + 0.1) * (1 + 0.03 * min(dt / life, 1)), flash=fl)
    elif kind == "year":
        put_text(fr, typed(TITLE[0], t, T["years"], 14), 540, 180, 130, BLACK)
        if t > T["years"] + 0.6:
            put_text(fr, typed(TITLE[1], t, T["years"] + 0.6, 22), 540, 295, 60, BLACK)
        place(fr, arg, 540, 600, 940, 440, s, flash=fl)
    elif kind == "price":
        place(fr, arg, 540, 620, 900, 420, s)
        if t > T.get("fraction", 1e9):
            p = pop(t, T["fraction"])
            put_text(fr, "FRACTION OF", 540, 170, 90, GREEN, 7, p)
            put_text(fr, "THE PRICE", 540, 270, 90, GREEN, 7, p)
    elif kind == "spec":
        place(fr, "a5_06_hoodopen", 300, 600, 500, 430, s)
        place(fr, "a5_07_engine", 800, 600, 480, 480, pop(t, t0 + 0.15) * (1 + 0.03 * dt / life))
        if t < T["hp208"]:
            put_text(fr, typed("EA888 2.0L TURBO", t, T["liter"], 24), 540, 175, 70, BLACK)
        else:
            put_text(fr, "208-220 HP", 540, 160, 96, BLUE, 0, pop(t, T["hp208"]))
            if t > T["lb258"]:
                put_text(fr, "258 LB-FT", 540, 265, 80, BLUE, 0, pop(t, T["lb258"]))
    elif kind == "feature":
        part, label, car = arg
        put_text(fr, label, 540, 175, 96, BLUE, 0, pop(t, t0))
        place(fr, part, 290, 580, 420, 420, pop(t, t0 + 0.08, 0.28, 0.3))
        place(fr, car, 760, 600, 560, 380, pop(t, t0 + 0.16) * (1 + 0.03 * dt / life))
    elif kind == "engine":
        k = min(int(dt / (ENG_DUR / len(ENG_SHOTS))), len(ENG_SHOTS) - 1); ct = t0 + k * ENG_DUR / len(ENG_SHOTS)
        # shake only while the engine sound is playing; it eases out with the sound's fade and then stops
        left = (ENG_START + ENG_DUR) - t
        amp = 9 * min(1.0, max(0.0, left / 0.4))
        sx, sy = shake(t, amp) if amp > 0 else (0, 0)
        zoom = 1 + 0.14 * (1 - ease_out((t - ct) / 0.3))
        place(fr, ENG_SHOTS[k], 540 + sx, CAR_Y + sy, 960, 520, zoom)
        put_text(fr, "AUDI A5 2.0T", 540, 140, 54, WHITE, 0, pop(t, t0 + 0.2), pill=True)
    elif kind == "mods":
        cur = [m for m in MODS if m[0] <= t]
        if not cur:
            place(fr, "a5_07_engine", 540, CAR_Y, 520, 520, s)
        else:
            mt, part, label = cur[-1]
            put_text(fr, label, 540, 175, 92, BLUE, 0, pop(t, mt))
            if part:
                place(fr, "a5_07_engine", 300, 560, 420, 420, 1.0)
                place(fr, part, 790, 560, 400, 400, pop(t, mt, 0.28, 0.3))
            else:   # no part picture: the engine itself, bigger
                place(fr, "a5_07_engine", 540, 580, 520, 520, pop(t, mt, 0.28, 0.6))
            # tray of collected parts, lower-left
            for j, (pt, pp, _) in enumerate([c for c in cur[:-1] if c[1]]):
                cx, cy = 175 + (j % 2) * 260, 1330 + (j // 2) * 260
                place(fr, pp, cx, cy, 210, 210, pop(t, pt + 0.6))
    elif kind == "hp":
        prog = ease_io((t - T["around"]) / max(T["hp_end"] - T["around"], 0.3))
        hp = int(220 + 70 * prog) if t > T["around"] else 220
        logo_w = 220
        tw = text_img(f"{hp}HP", 118, BLUE).width
        x0 = 540 - (logo_w + 30 + tw) / 2
        place(fr, LOGO, x0 + logo_w / 2, 210, logo_w, logo_w, pop(t, t0))
        hit = 1 + 0.25 * (1 - ease_out((t - T["hp_end"]) / 0.35)) if t > T["hp_end"] else 1
        put_text(fr, f"{hp}HP", x0 + logo_w + 30, 210, 118, BLUE, 0, pop(t, t0 + 0.08) * hit, anchor="l")
        place(fr, arg, 540, 620, 900, 420, s)

# ---------------------------------------------------------------- character
# Motion rules measured from the stick-arm reference (05_CHARACTER_MOTION_ANALYSIS.md), converted to 30 fps.
from num2words import num2words
import re as _re

def _spoken(w):
    w = w.lower()
    def num(m):
        x = m.group(0)
        if "." in x: return " point ".join(num2words(int(p)) for p in x.split("."))
        v = int(x)
        return num2words(v, to="year") if len(x) == 4 and 1900 < v < 2100 else num2words(v)
    return _re.sub(r"[^a-z]", "", _re.sub(r"\d+(?:\.\d+)?", num, w))

def _visemes(word):
    """letters -> mouth shapes: 1 closed (m b p), 2 teeth (s t d n l r k g ...), 3 open (a i), 4 small o (oo u w),
    5 two rows (ee ch sh j), 6 f/v, 7 round O (o aw)"""
    s = _spoken(word); out = []; i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in ("oo", "ou", "ow", "qu"): out.append(4); i += 2; continue
        if two in ("ee", "ea", "ch", "sh"): out.append(5); i += 2; continue
        if two in ("ph",): out.append(6); i += 2; continue
        if two in ("th",): out.append(2); i += 2; continue
        c = s[i]; i += 1
        if c in "mbp": out.append(1)
        elif c in "fv": out.append(6)
        elif c in "w": out.append(4)
        elif c in "u": out.append(4)
        elif c in "o": out.append(7)
        elif c in "ai": out.append(3)
        elif c in "ey": out.append(2)
        elif c in "j": out.append(5)
        else: out.append(2)
    # keep vowels + visible consonants, drop repeats
    keep = [v for k, v in enumerate(out) if k == 0 or v != out[k - 1]]
    return keep or [2]

# per-frame mouth track: each word's shapes spread over its spoken time, 2-4 frames per shape, closed in pauses
NF_ALL = int(TOTAL * FPS) + 2
MOUTH = np.full(NF_ALL, 2, int)          # rest mouth = the teeth grin (reference V2)
for w in WORDS:
    f0, f1 = int(round(w["S"] * FPS)), max(int(round(w["E"] * FPS)), int(round(w["S"] * FPS)) + 2)
    seq = _visemes(w["w"])
    n = max(1, min(len(seq), (f1 - f0) // 2))           # at least 2 frames per shape
    pick = [seq[int(k * len(seq) / n)] for k in range(n)]
    for k, v in enumerate(pick):
        a0 = f0 + (f1 - f0) * k // n; a1 = f0 + (f1 - f0) * (k + 1) // n
        MOUTH[a0:a1] = 8 if v in (2, 5) else v      # while talking: the smaller grin

def mouth_shape(t):
    return int(MOUTH[min(int(t * FPS), NF_ALL - 1)])

def _ev(word, n=1, dt=0.0):
    return at(word, n) + dt

DEFAULT = "rest"
MOVE = 2 / 30              # reference V2: poses snap in 2 frames, no in-between, no overshoot
HOLD = 3.0                 # reference V2: ~3.3 s per pose

G = lambda w, pose, n=1, hold=HOLD: (_ev(w, n) - 0.03, pose, hold)
GEST_EVENTS = sorted([      # confident, decisive poses on the key words (~every 3 s), no shrugs
    G("want", "raise_R"), G("beautiful", "open"), G("glance,", "out_R_hip"), G("talking", "raise_L_hip"),
    G("Because", "out_L"), G("underestimate", "hips"), G("They", "out_R"), G("But", "raise_R_hip"),
    G("EA888", "out_L_hip"), G("Depending", "open"), G("208", "raise_R"), G("potential", "hips"),
    G("mod", "out_R"), G("intake,", "out_L", hold=1.2), G("intercooler,", "raise_L", hold=1.2),
    G("downpipe,", "out_L", hold=1.2), G("Stage", "raise_L", hold=1.6), G("That's", "raise_R_hip"),
    G("with", "open", 2), G("280–290", "raise_L"), G("K04", "out_L_hip"), G("Suddenly,", "raise_R"),
    G("becomes", "open"), G("styling", "out_R_hip"), G("Basically,", "hips"), G("Would", "raise_R_hip"),
    G("RS", "open", 2, hold=TOTAL),
], key=lambda e: e[0])
TILTS = [2.0, -1.5, 2.5, -2.0, 1.0, -1.0, 2.0, -2.5]   # confident: steadier head

def _pose_vec(name):
    p = POSES[name]; return np.array([p["L"][0], p["L"][1], p["R"][0], p["R"][1]], float)

def arm_pose(t):
    cur = _pose_vec(DEFAULT)
    for k, (st, name, hold) in enumerate(GEST_EVENTS):
        if st > t: break
        nxt = GEST_EVENTS[k + 1][0] if k + 1 < len(GEST_EVENTS) else 1e9
        start = cur.copy(); x = min(max((t - st) / MOVE, 0.0), 1.0)
        cur = start + (_pose_vec(name) - start) * x
        end_hold = min(st + hold, nxt)
        if t > end_hold and nxt - end_hold > 0.4:          # snap back to rest if the next pose is far away
            y = min(max((t - end_hold) / MOVE, 0.0), 1.0)
            cur = cur + (_pose_vec(DEFAULT) - cur) * y
    return {"L": (cur[0], cur[1]), "R": (cur[2], cur[3])}

_rng = random.Random(11)
BLINKS = []
_t = 2.5
while _t < TOTAL:
    BLINKS.append(_t); _t += _rng.uniform(6.0, 9.0)         # reference V2: rare blinks (~every 8 s)

def head_state(t):
    # head tilts toward a new angle at each pose change and holds it; slow sideways drift; body never moves
    tilt, prev = 0.0, 0.0
    for k, (st, name, hold) in enumerate(GEST_EVENTS):
        if st > t: break
        tgt = TILTS[k % len(TILTS)]
        x = ease_io((t - st) / 0.45)
        tilt = prev + (tgt - prev) * x
        prev = tilt if x >= 1 else prev + (tgt - prev) * x
        prev = tgt if x >= 1 else prev
    dx = 9 * math.sin(2 * math.pi * t / 7.3)
    blink = any(0 <= t - b < 0.13 for b in BLINKS)
    return dx, -3.0, tilt, 0.0, blink                         # chin slightly up

def draw_char(fr, t):
    dx, dy, tilt, brow, blink = head_state(t)
    im = RIG.pose(arm_pose(t), mouth_shape(t), blink, brow, head_dx=dx, head_dy=dy, tilt=tilt, sway=0.25 * dx)
    a = ease_io((t - MOD_MODE[0]) / 0.35) - ease_io((t - MOD_MODE[1]) / 0.35)
    scale = (920 / RIG.h) * (1 - a) + (760 / RIG.h) * a      # by the character's own height (canvas has padding)
    cx = 540 * (1 - a) + 800 * a
    feet = 1905
    w, h = int(im.width * scale), int(im.height * scale)
    c = im.resize((w, h), Image.LANCZOS)
    fr.alpha_composite(c, (int(cx - w / 2), int(feet - h)))

# ---------------------------------------------------------------- captions
def current_word(t):
    for k, w in enumerate(WORDS):
        nxt = WORDS[k + 1]["S"] if k + 1 < len(WORDS) else w["E"] + 0.6
        end = min(nxt, w["E"] + 0.12)   # clear when the voice stops (or the next word starts)
        if w["S"] <= t < end:
            return w
    return None

def draw_caption(fr, t):
    w = current_word(t)
    if not w:
        return
    put_text(fr, w["w"], 540, 905, 74, WHITE, 8, pop(t, w["S"], 0.12, 0.75))     # captions 20% smaller

# ---------------------------------------------------------------- frame
def render(fi):
    t = fi / FPS
    fr = BG.copy().convert("RGBA")
    draw_scene(fr, t)
    draw_char(fr, t)
    draw_caption(fr, t)
    for ft in FLASHES:
        if 0 <= t - ft < 0.14:
            a = int(150 * (1 - (t - ft) / 0.14))
            fr.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, a)))
    return fr.convert("RGB").tobytes()

# ---------------------------------------------------------------- audio
SR = 44100
rng = np.random.default_rng(7)

def env(n, a, d):
    e = np.ones(n); na, nd = int(a * SR), int(d * SR)
    e[:na] = np.linspace(0, 1, na) if na else 1
    e[-nd:] *= np.linspace(1, 0, nd) ** 2
    return e

def lp(x, cut):  # one-pole low-pass with per-sample cutoff (Hz)
    y = np.zeros_like(x); a = 1 - np.exp(-2 * np.pi * np.broadcast_to(cut, x.shape) / SR); s = 0.0
    for i in range(len(x)):
        s += a[i] * (x[i] - s); y[i] = s
    return y

def sfx_whoosh(d=0.38):
    n = int(d * SR); x = rng.standard_normal(n)
    cut = 400 + 5000 * np.sin(np.linspace(0, np.pi, n)) ** 2
    y = lp(x, cut) - lp(x, cut * 0.25)
    return y / np.abs(y).max() * env(n, d * 0.45, d * 0.5)

def sfx_pop():
    n = int(0.09 * SR); t = np.arange(n) / SR
    f = 900 * np.exp(-t * 40) + 260
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)
    return y

def sfx_click():
    n = int(0.012 * SR); x = rng.standard_normal(n); y = x - lp(x, 2500)
    return y / np.abs(y).max() * np.exp(-np.arange(n) / SR * 400)

def sfx_boom():
    n = int(1.1 * SR); t = np.arange(n) / SR
    f = 95 * np.exp(-t * 5) + 42
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    return y / np.abs(y).max()          # pure tone (no noise layer)

def sfx_riser(d):
    n = int(d * SR); t = np.arange(n) / SR
    f = 180 * (7 ** (t / d))
    y = 0.35 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = rng.standard_normal(n); y += lp(x, 300 + 6000 * (t / d) ** 2) * 0.9
    y *= (t / d) ** 2
    return y / np.abs(y).max()

def sfx_ding():
    n = int(0.9 * SR); t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 1318.5 * t) + 0.7 * np.sin(2 * np.pi * 1975.5 * t)) * np.exp(-t * 5)
    y[: int(0.07 * SR)] += lp(rng.standard_normal(int(0.07 * SR)), 5000) * 1.5 * np.exp(-np.arange(int(0.07 * SR)) / SR * 60)
    return y / np.abs(y).max()

def sfx_shine():
    """metallic 'bling' rebuilt from the reference: partials 2040/2379/2832/3887/4441 Hz, two strikes, ~0.85 s ring"""
    d = 0.9; n = int(d * SR); t = np.arange(n) / SR
    parts = [(2040, .23), (2379, .55), (2832, .62), (3887, 1.0), (4441, .21)]
    y = np.zeros(n)
    for f, amp in parts:
        vib = 1 + 0.003 * np.sin(2 * np.pi * 6 * t)
        y += amp * np.sin(2 * np.pi * np.cumsum(f * vib) / SR)
    env = np.where(t < 0.2, 1.0, np.exp(-(t - 0.2) / 0.17))
    env *= np.minimum(t / 0.002, 1)
    env *= 1 - 0.45 * np.exp(-((t - 0.13) / 0.025) ** 2)   # small dip...
    env += 0.5 * np.exp(-np.maximum(t - 0.18, 0) / 0.08) * (t >= 0.18)  # ...second strike
    tr = rng.standard_normal(n) * np.exp(-t / 0.012); tr = tr - lp(tr, 5000)
    y = y * env + 0.25 * tr
    return y / np.abs(y).max()

def sfx_part():
    """parts / labels appear sound: tonal pop (pitch drop) + short bright tick -- pure tones, no noise"""
    n = int(0.22 * SR); t = np.arange(n) / SR
    f = 260 + 640 * np.exp(-t * 38)                       # 900 -> 260 Hz drop: the 'pop'
    pop = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 22) * np.minimum(t / 0.002, 1)
    tick = (np.sin(2 * np.pi * 2350 * t) + 0.5 * np.sin(2 * np.pi * 3520 * t)) * np.exp(-t * 60) * 0.45
    y = pop + tick
    return y / np.abs(y).max()

def sfx_swish():
    """soft tonal swipe for cuts: a quick pitch glide, no noise (no hiss)"""
    n = int(0.14 * SR); t = np.arange(n) / SR
    f = 260 + 900 * (t / t[-1]) ** 1.5
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 2
    return y / np.abs(y).max()

def build_audio(path):
    n = int(TOTAL * SR) + SR
    mix = np.zeros(n, np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        if i < 0 or i >= n: return
        seg = sig[: n - i]; mix[i:i + len(seg)] += (seg * gain).astype(np.float32)

    cut_i = int(HOOK_END_VO * SR)
    fd = int(0.01 * SR); hook = vo[:cut_i].copy(); hook[-fd:] *= np.linspace(1, 0, fd)
    add(hook, 0, 1.0)
    rest = vo[cut_i:].copy(); rest[:fd] *= np.linspace(0, 1, fd)
    add(rest, HOOK_END_VO + SHIFT, 1.0)
    if USE_ENGINE:
        eng, esr = sf.read(f"{P}/raw/engine_sound.wav", dtype="float32")
        eng = eng.mean(1) if eng.ndim > 1 else eng
        eng = eng[: int(ENG_DUR * SR)]
        eng = eng / np.abs(eng).max() * 0.9 * env(len(eng), 0.04, 0.4)
        add(eng, ENG_START, 1.0)

    ck = sfx_click()
    ap, _ = sf.read(f"{ROOT}/assets/sfx/appear_reference.wav", dtype="float32")
    ap_ratio = float(open(f"{ROOT}/assets/sfx/appear_reference.txt").read().split("= ")[-1].split()[0])
    _env = np.convolve(np.abs(vo), np.ones(441) / 441, "same")
    _speech = np.sqrt((vo[_env > 0.02] ** 2).mean())
    _ap_rms = np.sqrt((np.pad(ap, (0, max(0, int(0.4 * SR) - len(ap))))[: int(0.4 * SR)] ** 2).mean())
    ap_gain = ap_ratio * _speech / _ap_rms
    print(f"appear gain {ap_gain:.3f} (effect/speech RMS = {ap_ratio:.3f}, as in the reference)")
    for t0, txt, cps in ((T["years"], TITLE[0], 14), (T["years"] + 0.6, TITLE[1], 22), (T["liter"], "EA888 2.0L TURBO", 24)):
        for c in range(len(txt)):                       # typing sound: one key click per typed letter
            if txt[c] != " ":
                add(ck, t0 + (c + 1) / cps, 0.12 * random.uniform(0.75, 1.0))
    # the car appear sound (from the reference) only when a car shot appears
    for tt in [st for st, kind, _ in SCENES[1:] if kind != "engine"]:
        if not any(abs(tt - x) < 0.25 for x in SHINES):        # the shine has its own sound
            add(ap, tt - 0.015, ap_gain)
    # parts, numbers and labels: their own sound, never the car's
    pt = sfx_part()
    pt_gain = 0.85 * _speech / np.sqrt((pt[: int(0.1 * SR)] ** 2).mean())
    for tt in [T["hp208"], T["lb258"], T["k04"]] + [m[0] for m in MODS]:
        add(pt, tt, pt_gain)
    # the reference's own shine sound, placed on the first flash frame, at the reference's level relative to speech
    sh, ssr = sf.read(f"{ROOT}/assets/sfx/shine_reference.wav", dtype="float32")
    if sh.ndim > 1: sh = sh.mean(1)
    ratio = float(open(f"{ROOT}/assets/sfx/shine_reference.txt").read().split("= ")[-1].split()[0])
    vo_act = vo[np.abs(vo) > 0.01]
    speech_rms = np.sqrt((vo[(np.convolve(np.abs(vo), np.ones(441) / 441, "same") > 0.02)] ** 2).mean())
    shine_rms = np.sqrt((sh[: int(0.6 * SR)] ** 2).mean())
    g = ratio * speech_rms / shine_rms
    print(f"shine gain {g:.3f} (effect/speech RMS = {ratio:.3f}, as in the reference)")
    for st in SHINES:
        add(sh, st, g)
    peak = np.abs(mix).max()
    if peak > 0.98: mix *= 0.98 / peak
    sf.write(path, mix[: int(TOTAL * SR)], SR)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{P}/Audi_A5_2.0T.mp4"
    preview = os.environ.get("PREVIEW")
    if MISSING: print("placeholders for missing images:", MISSING)
    if preview:   # render a few still frames
        for s in preview.split(","):
            t = float(s); Image.frombytes("RGB", (W, H), render(int(t * FPS))).save(f"{BUILD}/prev_{s}.png")
        sys.exit()
    build_audio(f"{BUILD}/mix.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{BUILD}/mix.wav", "-af", "loudnorm=I=-14:TP=-1:LRA=11",
                    "-ar", "44100", f"{BUILD}/mix_norm.wav"], check=True)
    nf = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", f"{BUILD}/mix_norm.wav", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for k, b in enumerate(pool.imap(render, range(nf), chunksize=8)):
            ff.stdin.write(b)
            if k % 300 == 0: print(f"frame {k}/{nf}", flush=True)
    ff.stdin.close(); ff.wait()
    print("done", out, f"{TOTAL:.1f}s")
