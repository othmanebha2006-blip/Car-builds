"""Render the BMW 550i F10 short (1080x1920, 30 fps) from cut-out assets, voiceover, engine clip and word timings."""
import json, math, os, random, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/user/Car-builds"
P = f"{ROOT}/projects/550i"
CUT, BUILD = f"{P}/cut", f"{P}/build"
FONT = f"{ROOT}/tools/fonts/Montserrat-900.ttf"
W, H, FPS = 1080, 1920, 30
BLUE, GREEN, BLACK, WHITE = (30, 144, 255), (34, 197, 94), (12, 12, 12), (255, 255, 255)

# ---------------------------------------------------------------- timeline
HOOK_END_VO = 7.25          # voiceover time where the hook ends
ENG_START, ENG_DUR = 7.30, 5.20
SHIFT = ENG_START + ENG_DUR + 0.12 - 7.32   # vo time 7.32 -> just after engine clip
VO_DUR = sf.info(f"{BUILD}/vo_tight.wav").duration
TOTAL = VO_DUR + SHIFT + 1.4

def vt(t):  # voiceover time -> final time
    return t if t < HOOK_END_VO else t + SHIFT

WORDS = json.load(open(f"{BUILD}/words.json"))
for w in WORDS:
    w["S"], w["E"] = vt(w["s"]), vt(w["e"])

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
def load(name):
    return Image.open(f"{CUT}/{name}.png").convert("RGBA")

IMG = {n: load(n) for n in [
    "f10_01_front34", "f10_02_rear34", "f10_03_side", "f10_04_front", "f10_05_rear", "f10_06_hoodopen",
    "f10_07_engine", "f10_08_front34_low", "part_downpipe", "part_intake", "part_heatexchanger", "part_tune", "logo_bmw"]}
BG = Image.open(f"{P}/raw/background.png").convert("RGB").resize((W, H))

# character variants: mouth 0..3 x blink
sys.path.insert(0, f"{ROOT}/tools")
from char_rig import Rig, GESTURES
RIG = Rig(f"{CUT}/character2.png")

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

def place(frame, name, cx, cy, bw, bh, scale=1.0, alpha=1.0):
    im = IMG[name]; s = fit(im, bw, bh) * scale
    w, h = int(im.width * s), int(im.height * s)
    if w < 2 or h < 2:
        return
    # quantise sizes to keep the cache effective
    w -= w % 2; h -= h % 2
    sm = scaled(name, w, h)
    if alpha < 1:
        sm = sm.copy(); sm.putalpha(sm.getchannel("A").point(lambda a: int(a * alpha)))
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
T = dict(
    underrated=at("underrated"), hidden=at("hidden"), used=at("used"),
    found=at("foundation") - 0.1, years=at("2014"), since=at("Since"), most=at("most", 2), youcan=at("You", 2),
    fraction=at("fraction"), 
    liter=at("4.4"), n63=at("N63"), hp445=at("445"), lb480=at("480"), straight=at("straight"), serious=at("serious"),
    now=at("Now"), crazy=at("crazy"), start=at("Start"), down=at("downpipes"), intake=at("intake"),
    cooling=at("cooling"), ecu=at("ECU"), result=at("result"), around=at("Around"), hp550_end=at("horsepower", 2, end=True),
    sudden=at("Suddenly"), machine=at("machine"), pulls=at("pulls"), delivers=at("delivers"), surprise=at("surprise"),
    basically=at("basically"), so=at("So"), actual=at("actual"),
)
# "and under the hood" -> first "and" after "price."
T["under"] = next(w["S"] for w in WORDS if w["w"] == "and" and w["S"] > T["fraction"])

SCENES = [  # (start, kind, args)
    (0.0, "logo_car", "f10_01_front34"),
    (T["underrated"], "car", "f10_08_front34_low"),
    (T["hidden"], "car", "f10_02_rear34"),
    (T["used"], "car", "f10_03_side"),
    (ENG_START, "engine", None),
    (T["found"], "year", "f10_03_side"),
    (T["since"], "diag", ("f10_02_rear34", "f10_01_front34")),
    (T["most"], "car", "f10_04_front"),
    (T["youcan"], "price", "f10_05_rear"),
    (T["under"], "spec", None),
    (T["straight"], "car", "f10_08_front34_low"),
    (T["serious"], "logo_car", "f10_01_front34"),
    (T["now"], "car", "f10_04_front"),
    (T["start"], "mods", None),
    (T["result"], "hp", "f10_08_front34_low"),
    (T["sudden"], "diag", ("f10_05_rear", "f10_08_front34_low")),
    (T["machine"], "car", "f10_01_front34"),
    (T["pulls"], "car", "f10_02_rear34"),
    (T["delivers"], "car", "f10_05_rear"),
    (T["surprise"], "car", "f10_08_front34_low"),
    (T["basically"], "logo_car", "f10_03_side"),
    (T["so"], "logo_car", "f10_01_front34"),
]
SCENE_T = [s[0] for s in SCENES]
MODS = [(T["down"], "part_downpipe", "DOWNPIPES"), (T["intake"], "part_intake", "INTAKE"),
        (T["cooling"], "part_heatexchanger", "COOLING"), (T["ecu"], "part_tune", "ECU TUNE")]
MOD_MODE = (T["start"] - 0.15, T["result"] - 0.1)
FLASHES = [ENG_START, T["crazy"] + 0.25, T["hp550_end"], T["so"]]
ENG_SHOTS = ["f10_08_front34_low", "f10_05_rear", "f10_04_front", "f10_02_rear34"]

def shake(t, amp, seed=0):
    r = random.Random(int(t * FPS) * 7 + seed)
    return r.uniform(-amp, amp), r.uniform(-amp, amp)

def draw_scene(fr, t):
    i = max(k for k, s in enumerate(SCENE_T) if s <= t)
    t0, kind, arg = SCENES[i]
    nxt = SCENE_T[i + 1] if i + 1 < len(SCENE_T) else TOTAL
    dt, life = t - t0, max(nxt - t0, 0.1)
    s = pop(t, t0) * (1 + 0.035 * min(dt / life, 1))   # pop-in + slow push-in
    if kind == "car":
        place(fr, arg, 540, CAR_Y, 960, 520, s)
    elif kind == "logo_car":
        place(fr, "logo_bmw", 540, 215, 250, 250, pop(t, t0 + 0.05))
        place(fr, arg, 540, 590, 900, 420, s)
    elif kind == "diag":
        place(fr, arg[0], 330, 300, 560, 300, pop(t, t0))
        place(fr, arg[1], 660, 640, 760, 400, pop(t, t0 + 0.12) * (1 + 0.03 * dt / life))
    elif kind == "year":
        put_text(fr, typed("2014-2016", t, T["years"], 14), 540, 180, 130, BLACK)
        if t > T["years"] + 0.45:
            put_text(fr, typed("BMW 550I F10", t, T["years"] + 0.45, 22), 540, 300, 56, BLACK)
        place(fr, arg, 540, 600, 940, 440, s)
    elif kind == "price":
        place(fr, arg, 540, 600, 860, 420, s)
        if t > T["fraction"]:
            p = pop(t, T["fraction"])
            put_text(fr, "FRACTION OF", 540, 170, 78, GREEN, 6, p)
            put_text(fr, "THE M5 PRICE", 540, 265, 78, GREEN, 6, p)
    elif kind == "spec":
        place(fr, "f10_06_hoodopen", 300, 590, 500, 430, s)
        place(fr, "f10_07_engine", 790, 590, 440, 440, pop(t, t0 + 0.15) * (1 + 0.03 * dt / life))
        if t < T["hp445"]:
            if t > T["liter"]:
                put_text(fr, typed("4.4L TWIN-TURBO V8", t, T["liter"], 24), 540, 175, 70, BLACK)
            if t > T["n63"]:
                put_text(fr, "N63", 540, 275, 66, BLUE, 0, pop(t, T["n63"]))
        else:
            put_text(fr, "445 HP", 540, 175, 100, BLUE, 0, pop(t, T["hp445"]))
            if t > T["lb480"]:
                put_text(fr, "480 LB-FT", 540, 285, 80, BLUE, 0, pop(t, T["lb480"]))
    elif kind == "engine":
        k = min(int(dt / (ENG_DUR / 4)), 3); ct = t0 + k * ENG_DUR / 4
        sx, sy = shake(t, 9)
        zoom = 1 + 0.14 * (1 - ease_out((t - ct) / 0.3))
        place(fr, ENG_SHOTS[k], 540 + sx, CAR_Y + sy, 960, 520, zoom)
        put_text(fr, "BMW 550I F10", 540, 140, 54, WHITE, 0, pop(t, t0 + 0.2), pill=True)
    elif kind == "mods":
        cur = [m for m in MODS if m[0] <= t]
        if not cur:
            place(fr, "f10_07_engine", 540, CAR_Y, 520, 520, s)
        else:
            mt, part, label = cur[-1]
            put_text(fr, label, 540, 175, 92, BLUE, 0, pop(t, mt))
            place(fr, "f10_07_engine", 300, 560, 420, 420, 1.0)
            place(fr, part, 790, 560, 400, 400, pop(t, mt, 0.28, 0.3))
            # tray of collected parts, lower-left
            for j, (pt, pp, _) in enumerate(cur[:-1]):
                cx, cy = 175 + (j % 2) * 260, 1330 + (j // 2) * 260
                place(fr, pp, cx, cy, 210, 210, pop(t, cur[j + 1][0] + 0.05))
    elif kind == "hp":
        prog = ease_io((t - T["around"]) / max(T["hp550_end"] - T["around"], 0.3))
        hp = int(445 + 105 * prog) if t > T["around"] else 445
        logo_w = 150
        tw = text_img(f"{hp}HP", 118, BLUE).width
        x0 = 540 - (logo_w + 30 + tw) / 2
        place(fr, "logo_bmw", x0 + logo_w / 2, 210, logo_w, logo_w, pop(t, t0))
        hit = 1 + 0.25 * (1 - ease_out((t - T["hp550_end"]) / 0.35)) if t > T["hp550_end"] else 1
        put_text(fr, f"{hp}HP", x0 + logo_w + 30, 210, 118, BLUE, 0, pop(t, t0 + 0.08) * hit, anchor="l")
        place(fr, arg, 540, 620, 900, 420, s)

# ---------------------------------------------------------------- character
def _phrases():
    """group words into phrases split at pauses / sentence ends"""
    ph, cur = [], []
    for k, w in enumerate(WORDS):
        cur.append(w)
        nxt = WORDS[k + 1] if k + 1 < len(WORDS) else None
        if nxt is None or nxt["S"] - w["E"] > 0.12 or w["w"][-1] in ".,?!":
            ph.append((cur[0]["S"], cur[-1]["E"], " ".join(x["w"] for x in cur))); cur = []
    return ph

PHRASES = _phrases()
CYCLE = ["open_small", "out_r", "open", "out_l", "open_small", "out_r", "rest", "out_l"]
MOD_CYCLE = ["out_l", "open_small", "point_up", "open_small"]

def _gesture_keys():
    keys, ci = [(-1.0, "rest")], 0
    for k, (s, e, txt) in enumerate(PHRASES):
        low = txt.lower()
        if k == 0 or low.startswith("now,") or low.startswith("so,"):
            g = "point_up"
        elif MOD_MODE[0] <= s < MOD_MODE[1]:
            g = MOD_CYCLE[ci % len(MOD_CYCLE)]; ci += 1
        else:
            g = CYCLE[ci % len(CYCLE)]; ci += 1
        keys.append((max(s - 0.18, 0.0), g))
        nxt = PHRASES[k + 1][0] if k + 1 < len(PHRASES) else TOTAL
        if nxt - e > 0.45:
            keys.append((e + 0.1, "rest"))
    keys.append((ENG_START, "rest"))
    return sorted(keys)

GKEYS = _gesture_keys()
# head tilt targets: change every couple of phrases, small angles
TILTS = [(0.0, 0.0)] + [(PHRASES[k][0] - 0.2, [2.2, -1.6, 0.0, -2.4, 1.4, 0.0][(k // 2) % 6]) for k in range(0, len(PHRASES), 2)]

def _interp(keys, t, dur, getter):
    prev = getter(keys[0][1]); cur = prev
    for kt, val in keys:
        if kt > t: break
        x = ease_io((t - kt) / dur)
        tgt = getter(val)
        cur = tuple(p + (q - p) * x for p, q in zip(prev, tgt)) if isinstance(tgt, tuple) else prev + (tgt - prev) * x
        prev = cur
    return cur

def char_state(t):
    # walk the keyframes so each transition starts from wherever the previous one was
    pose = (0.0, 0.0)
    for kt, g in GKEYS:
        if kt > t: break
        tgt = GESTURES[g]; x = ease_io((t - kt) / 0.32)
        pose = (pose[0] + (tgt[0] - pose[0]) * x, pose[1] + (tgt[1] - pose[1]) * x) if x < 1 else tgt
    tilt = 0.0
    for kt, a in TILTS:
        if kt > t: break
        x = ease_io((t - kt) / 0.7); tilt = tilt + (a - tilt) * x if x < 1 else a
    return pose, tilt

def draw_char(fr, t):
    lvl = mouth_level(t)
    blink = (t % 3.7) < 0.12 or (t % 11.3) < 0.1
    (la, ra), tilt = char_state(t)
    talking = 1.0 if lvl > 0 else 0.0
    # hands keep moving a little while he talks (explaining motion)
    la += talking * 6 * math.sin(t * 2 * math.pi * 0.9)
    ra += talking * 6 * math.sin(t * 2 * math.pi * 0.9 + 1.7)
    tilt += 0.6 * math.sin(t * 1.1)
    im = RIG.pose(la, ra, tilt, lvl, blink)
    # normal vs. corner mode (during mods)
    a = ease_io((t - MOD_MODE[0]) / 0.35) - ease_io((t - MOD_MODE[1]) / 0.35)
    scale = (920 / im.height) * (1 - a) + (780 / im.height) * a
    cx = 540 * (1 - a) + 850 * a
    feet = 1905
    w, h = int(im.width * scale), int(im.height * scale)
    c = im.resize((w, h), Image.LANCZOS)
    fr.alpha_composite(c, (int(cx - w / 2), int(feet - h)))

# ---------------------------------------------------------------- captions
def current_word(t):
    for k, w in enumerate(WORDS):
        nxt = WORDS[k + 1]["S"] if k + 1 < len(WORDS) else w["E"] + 0.6
        end = min(nxt, w["E"] + 0.6)
        if w["S"] <= t < end:
            return w
    return None

def draw_caption(fr, t):
    w = current_word(t)
    if not w:
        return
    put_text(fr, w["w"], 540, 905, 92, WHITE, 10, pop(t, w["S"], 0.12, 0.75))

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
    y += 0.6 * lp(rng.standard_normal(n), 900) * np.exp(-t * 18)
    return y / np.abs(y).max()

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

def build_audio(path):
    n = int(TOTAL * SR) + SR
    mix = np.zeros(n, np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        if i < 0 or i >= n: return
        seg = sig[: n - i]; mix[i:i + len(seg)] += (seg * gain).astype(np.float32)

    a = vo if VSR == SR else None
    cut_i = int(HOOK_END_VO * SR)
    add(a[:cut_i], 0, 1.0)
    add(a[int(7.32 * SR) - 1500:], vt(7.32) - 1500 / SR, 1.0)
    eng, esr = sf.read(f"{P}/raw/engine_sound.wav", dtype="float32")
    eng = eng.mean(1) if eng.ndim > 1 else eng
    eng = eng[: int(ENG_DUR * SR)]
    eng = eng / np.abs(eng).max() * 0.9 * env(len(eng), 0.04, 0.45)
    add(eng, ENG_START, 1.0)

    wh, pp, ck, bm, dg = sfx_whoosh(), sfx_pop(), sfx_click(), sfx_boom(), sfx_ding()
    for st, kind, _ in SCENES[1:]:
        if kind != "engine":
            add(wh, st - 0.12, 0.16)
    for k in range(1, 4):
        add(wh, ENG_START + k * ENG_DUR / 4 - 0.1, 0.10)
    for tt in (T["years"], T["n63"], T["hp445"], T["lb480"], T["fraction"], ENG_START + 0.2):
        add(pp, tt, 0.28)
    for (mt, _, _) in MODS:
        add(pp, mt, 0.32); add(wh, mt - 0.1, 0.14)
    for t0, txt, cps in ((T["years"], "2014-2016", 14), (T["years"] + 0.45, "BMW 550I F10", 22), (T["liter"], "4.4L TWIN-TURBO V8", 24)):
        for c in range(len(txt)):
            add(ck, t0 + c / cps, 0.10 * random.uniform(0.7, 1.0))
    add(dg, T["fraction"] + 0.05, 0.22)
    rd = 1.6; add(sfx_riser(rd), T["crazy"] + 0.25 - rd, 0.16)
    add(bm, T["crazy"] + 0.25, 0.55)
    add(sfx_riser(T["hp550_end"] - T["around"]), T["around"], 0.10)
    add(bm, T["hp550_end"], 0.6)
    add(wh, MOD_MODE[0], 0.18); add(wh, MOD_MODE[1], 0.18)
    add(bm, T["so"], 0.3)
    peak = np.abs(mix).max()
    if peak > 0.98: mix *= 0.98 / peak
    sf.write(path, mix[: int(TOTAL * SR)], SR)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{P}/BMW_550i_F10.mp4"
    preview = os.environ.get("PREVIEW")
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
