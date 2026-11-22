"""Rig for the stick-arm 'Mecmation' character (projects/rs5/cut/character.png, 632x1400).

Arms are redrawn every frame as smooth curved strokes (shoulder -> elbow -> wrist) in the character's own line
width, with the original hand drawing attached at the wrist.  Head (beanie + face) is a separate layer that can
drift/tilt; mouth uses 7 phoneme shapes (visemes) drawn in the character's style; eyes blink; brows lift.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PAD = 330            # side padding so raised arms stay on the canvas
TOP = 90             # top padding: head tilt / body lean must never push the hat off the canvas
SS = 2               # supersampling for strokes
MOUTH_SCALE = 1.0    # set < 1 for a smaller talking mouth

CFG = dict(
    stroke=20, outline=17, skin=(253, 207, 164), black=(0, 0, 0),
    mouth_c=(325, 452), mouth_clear=(232, 424, 418, 482), smile_box=(238, 428, 412, 478),
    eyes=[(262, 360), (365, 360)], eye_r=(17, 23),
    brows=[(230, 276, 308, 318), (339, 276, 418, 318)],
    arms={"L": dict(A=(172, 650), W=(59, 845), erase=[(0, 652), (159, 652), (152, 690), (145, 720), (140, 760), (128, 790),
                                                    (106, 800), (106, 860), (100, 900), (105, 965), (0, 965)]),
          "R": dict(A=(454, 650), W=(573, 845), erase=[(632, 652), (473, 652), (480, 690), (487, 720), (492, 760), (504, 790),
                                                      (526, 800), (526, 860), (532, 900), (527, 965), (632, 965)])},
    wrist_cut=842,
    lean_radius=600,     # head drift (px) is produced by leaning the whole upper body about the hips
    leg_overlap=50,      # legs extend this far up under the shirt so the waist never opens
)

def _poly_mask(size, poly):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).polygon(poly, fill=255); return np.asarray(m) > 0

def _rot(v, deg):
    """rotate vector v by deg, counter-clockwise as seen on screen (y down)"""
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)

def _largest(mask):
    lab, n = ndimage.label(mask)
    if not n: return mask
    return lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))

class StickRig:
    """Layers are cut along the character's own outlines (not along straight rows):
       legs (extended up under the shirt) < torso (shirt + its own outline) < arms (strokes + original hands) < head.
       Torso, arms and head form one rigid upper body that leans about the hips; the head only adds a small tilt/nod
       about the chin on top of that, so neck and shoulders never separate."""
    def __init__(self, path, cfg=CFG):
        self.c = cfg
        src = Image.open(path).convert("RGBA")
        self.w, self.h = src.size
        cv = Image.new("RGBA", (self.w + 2 * PAD, self.h + TOP), (0, 0, 0, 0)); cv.paste(src, (PAD, TOP))
        A = np.array(cv); size = cv.size
        alpha = A[..., 3] > 0; rgb = A[..., :3].astype(int)
        Y = np.arange(size[1])[:, None] * np.ones((1, size[0]), int)
        P = lambda pts: [(x + PAD, y + TOP) for x, y in pts]
        self.P = lambda p: (p[0] + PAD, p[1] + TOP)
        ow = cfg["outline"]
        dark = (rgb.max(-1) < 90) & alpha

        # hands: original drawings below the wrist (unchanged)
        self.hands = {}; hand_any = np.zeros_like(alpha)
        for side, g in cfg["arms"].items():
            m = _poly_mask(size, P(g["erase"])) & alpha & (Y >= cfg["wrist_cut"] + TOP)
            hand = np.zeros_like(A); hand[m] = A[m]
            for k in range(6):
                r = hand[cfg["wrist_cut"] + TOP + k]; r[..., 3] = (r[..., 3] * (k + 1) / 7).astype(np.uint8)
            self.hands[side] = Image.fromarray(hand); hand_any |= m

        # head: face skin + beanie (+ text, eyes, mouth inside), plus its own outline ring
        skin_c = np.array(cfg["skin"])
        skin = (np.abs(rgb - skin_c).sum(-1) < 60) & alpha
        cx, cy = self.P(cfg["mouth_c"])
        lab, _ = ndimage.label(skin); face = lab == lab[cy - 60, cx]
        grey = (np.abs(rgb[..., 0] - rgb[..., 1]) < 14) & (np.abs(rgb[..., 1] - rgb[..., 2]) < 14) & (rgb.max(-1) > 70) & (rgb.max(-1) < 200) & alpha
        beanie = _largest(grey & (Y < TOP + 360))
        hat = alpha & (Y < TOP + cfg.get("hat_bottom", 330))          # everything this high is the beanie
        head_fill = ndimage.binary_fill_holes(ndimage.binary_closing(face | beanie | hat, iterations=6))
        head_fill = ndimage.binary_fill_holes(head_fill | (ndimage.binary_dilation(head_fill, iterations=ow + 2) & dark & (Y < TOP + 360)))
        d_head = ndimage.distance_transform_edt(~head_fill)
        head_m = (head_fill | (dark & (d_head <= ow + 3))) & alpha

        # torso: shirt fill (logo included) + original outline pixels in its band + regenerated ring where the
        # original outline was shared with the arms or hidden under the head
        red = _largest((rgb[..., 0] > 170) & (rgb[..., 1] < 110) & (rgb[..., 2] < 110) & alpha)
        torso_fill = ndimage.binary_fill_holes(ndimage.binary_closing(red, iterations=4))
        d_t = ndimage.distance_transform_edt(~torso_fill)
        band = (d_t > 0) & (d_t <= ow + 1)
        torso = np.zeros_like(A)
        torso[torso_fill] = A[torso_fill]
        keep = band & dark & ~head_m
        # the shirt's bottom edge is drawn thicker (~20 px) and anti-aliased: its whole outline belongs to the shirt
        tb = int(np.flatnonzero(torso_fill.any(1))[-1])
        hem = (d_t > 0) & (d_t <= ow + 7) & alpha & (Y > tb - 70) & ~head_m
        keep |= hem
        # shoulders: the drawn arm used to merge into the shirt outline here, so the cut-out outline was too thick
        # and stair-stepped. Keep only its inner part and let the anti-aliased ring give the smooth outer contour.
        shoulder = (Y > TOP + 530) & (Y < TOP + 720) & ~hem
        keep &= ~(shoulder & (d_t > ow - 2))
        torso[keep] = A[keep]
        ring_a = np.clip(ow + 0.5 - d_t, 0, 1) * (d_t > 0)
        ring = (ring_a > 0) & ~keep
        torso[ring, :3] = 0
        torso[ring, 3] = np.maximum(torso[ring, 3], (ring_a[ring] * 255).astype(np.uint8))
        self.torso_bottom = int(np.flatnonzero(torso_fill.any(1))[-1])

        # legs: everything left below the shirt (strokes + shoes), extended upward under the shirt
        legs_m = alpha & ~head_m & ~torso_fill & ~keep & ~(d_t <= ow + 1) & ~hand_any & (Y > self.torso_bottom - 40)
        legs = np.zeros_like(A); legs[legs_m] = A[legs_m]
        for x in np.flatnonzero(legs_m.any(0)):
            y0 = int(np.flatnonzero(legs_m[:, x])[0])
            if y0 > self.torso_bottom + 30: continue            # only columns that touch the shirt
            lo = max(0, y0 - cfg["leg_overlap"])
            legs[lo:y0, x] = legs[y0 + 2, x]
        self.legs = Image.fromarray(legs)
        self.body = Image.fromarray(torso)
        self.hip_pivot = (PAD + self.w / 2, self.torso_bottom - 10)

        # face parts: brows as sprites, eyes/mouth areas cleaned to skin
        head = np.zeros_like(A); head[head_m] = A[head_m]
        skin_t = tuple(cfg["skin"])
        self.brows = []
        for (x0, y0, x1, y1) in cfg["brows"]:
            reg = head[y0 + TOP:y1 + TOP, x0 + PAD:x1 + PAD]
            ink = (reg[..., :3].astype(int).sum(-1) < sum(skin_t) - 25) & (reg[..., 3] > 0)
            spr = np.zeros_like(reg); spr[ink] = reg[ink]
            self.brows.append(((x0 + PAD, y0 + TOP), Image.fromarray(spr)))
            reg[ndimage.binary_dilation(ink, iterations=3) & (reg[..., 3] > 0)] = skin_t + (255,)
        x0, y0, x1, y1 = cfg["smile_box"]
        self.smile = Image.fromarray(head[y0 + TOP:y1 + TOP, x0 + PAD:x1 + PAD].copy())
        reg = head[y0 + TOP:y1 + TOP, x0 + PAD:x1 + PAD]; reg[reg[..., 3] > 0] = skin_t + (255,)
        self.head = Image.fromarray(head)
        hb = int(np.flatnonzero(head_fill[:, int(cx)])[-1])
        self.head_pivot = (cx, hb - 15)                      # chin: head tilts/nods about the neck
        self.masks = dict(head=head_m, torso=torso_fill, legs=legs_m, hands=hand_any, alpha=alpha)
        self._face_cache = {}

    # ------------------------------------------------------------------ face
    def _mouth(self, d, v):
        cx, cy = self.P(self.c["mouth_c"]); K = (0, 0, 0)
        hw = 68
        def band(top_mid, top_end, bot_mid, n=24):
            up = [(cx - hw + 2 * hw * i / n, top_end + (top_mid - top_end) * (1 - ((2 * i / n) - 1) ** 2)) for i in range(n + 1)]
            lo = [(cx + hw - 2 * hw * i / n, top_end + (bot_mid - top_end) * (1 - ((2 * i / n) - 1) ** 2)) for i in range(n + 1)]
            return up, lo
        if v == 2:      # teeth smile (E, S, T, D, N ...)
            up, lo = band(cy - 4, cy - 14, cy + 22)
            d.polygon(up + lo, fill=(255, 255, 255), outline=K); d.line(up + lo + [up[0]], fill=K, width=10, joint="curve")
        elif v == 3:    # open with top teeth + tongue (A, I, open vowels)
            up, lo = band(cy - 2, cy - 16, cy + 44)
            d.polygon(up + lo, fill=(70, 12, 18))
            t_up, t_lo = band(cy - 2, cy - 16, cy + 10)
            d.polygon(t_up + t_lo, fill=(255, 255, 255))
            d.ellipse((cx - 30, cy + 18, cx + 30, cy + 46), fill=(214, 40, 52))
            d.line(up + lo + [up[0]], fill=K, width=10, joint="curve")
        elif v == 4:    # small 'o' (O, U, W, OO)
            d.ellipse((cx - 13, cy - 14, cx + 13, cy + 18), fill=K)
        elif v == 5:    # two rows of teeth (EE, CH, SH, J)
            up, lo = band(cy - 4, cy - 14, cy + 34)
            d.polygon(up + lo, fill=(255, 255, 255))
            mid = [(x, y + 13) for x, y in band(cy - 2, cy - 12, cy - 2)[0]]
            d.line(mid[2:-2], fill=K, width=8, joint="curve")
            d.line(up + lo + [up[0]], fill=K, width=10, joint="curve")
        elif v == 6:    # F / V: upper lip line with bottom teeth
            up, lo = band(cy + 2, cy - 10, cy + 2)
            d.line(up, fill=K, width=10, joint="curve")
            for i in range(6):
                x = cx - 45 + i * 18
                d.rounded_rectangle((x, cy + 4, x + 15, cy + 18), radius=5, fill=(255, 255, 255), outline=K, width=4)
        elif v == 7:    # round open 'O' (OH, AW)
            d.ellipse((cx - 26, cy - 14, cx + 26, cy + 34), fill=(70, 12, 18), outline=K, width=9)
            d.chord((cx - 17, cy + 10, cx + 17, cy + 30), 0, 180, fill=(214, 40, 52))
        # v == 1 (closed) handled by pasting the original smile

    def face(self, viseme=1, blink=False, brow=0.0):
        if viseme in (6, 8): viseme = 2              # F/V and "small grin" -> the clean teeth smile (the boxed-teeth F/V read as dots)
        key = (viseme, blink, round(brow, 1))
        if key in self._face_cache: return self._face_cache[key]
        im = self.head.copy(); d = ImageDraw.Draw(im)
        if viseme == 1:
            x0, y0, _, _ = self.c["smile_box"]; im.alpha_composite(self.smile, (x0 + PAD, y0 + TOP))
        else:
            m = Image.new("RGBA", (im.width * SS, im.height * SS), (0, 0, 0, 0))
            # draw mouth supersampled in a local patch for clean edges
            cx, cy = self.P(self.c["mouth_c"])
            patch = Image.new("RGBA", (200 * SS, 120 * SS), (0, 0, 0, 0)); pd = ImageDraw.Draw(patch)
            class _D:   # adapter: draw into the patch with local, scaled coordinates
                def __init__(s, dd): s.dd = dd
                def _t(s, pts): return [((x - cx + 100) * SS, (y - cy + 50) * SS) for x, y in pts]
                def polygon(s, pts, **k): s.dd.polygon(s._t(pts), fill=k.get("fill"))
                def line(s, pts, fill, width, joint=None): s.dd.line(s._t(pts), fill=fill, width=width * SS, joint="curve")
                def ellipse(s, b, fill=None, outline=None, width=1):
                    s.dd.ellipse(((b[0] - cx + 100) * SS, (b[1] - cy + 50) * SS, (b[2] - cx + 100) * SS, (b[3] - cy + 50) * SS), fill=fill, outline=outline, width=width * SS)
                def chord(s, b, a0, a1, fill=None):
                    s.dd.chord(((b[0] - cx + 100) * SS, (b[1] - cy + 50) * SS, (b[2] - cx + 100) * SS, (b[3] - cy + 50) * SS), a0, a1, fill=fill)
                def rounded_rectangle(s, b, radius, fill=None, outline=None, width=1):
                    s.dd.rounded_rectangle(((b[0] - cx + 100) * SS, (b[1] - cy + 50) * SS, (b[2] - cx + 100) * SS, (b[3] - cy + 50) * SS), radius=radius * SS, fill=fill, outline=outline, width=width * SS)
            self._mouth(_D(pd), viseme)
            k = MOUTH_SCALE                               # calm, smaller talking mouth
            patch = patch.resize((int(200 * k), int(120 * k)), Image.LANCZOS)
            im.alpha_composite(patch, (int(cx - 100 * k), int(cy - 50 * k)))
        if blink:
            for ex, ey in self.c["eyes"]:
                ex += PAD; ey += TOP; rx, ry = self.c["eye_r"]
                d.ellipse((ex - rx - 6, ey - ry - 6, ex + rx + 6, ey + ry + 6), fill=self.c["skin"])
                d.arc((ex - rx - 4, ey - 8, ex + rx + 4, ey + 14), 200, 340, fill=(0, 0, 0), width=8)
        for (pos, spr) in self.brows:
            im.alpha_composite(spr, (pos[0], int(round(pos[1] - brow))))
        self._face_cache[key] = im
        return im

    # ------------------------------------------------------------------ arms
    def _arm_points(self, side, out, bend):
        """out: upper-arm swing away from the body (deg); bend: elbow bend (deg, + folds the forearm up/inward)."""
        g = self.c["arms"][side]; A = self.P(g["A"]); W = self.P(g["W"])
        sgn = -1 if side == "L" else 1          # screen rotation sign for 'outward' on this side
        v = (W[0] - A[0], W[1] - A[1]); L = math.hypot(*v); L1 = L2 = L / 2
        u = _rot((v[0] / L, v[1] / L), sgn * out)
        E = (A[0] + u[0] * L1, A[1] + u[1] * L1)
        f = _rot(u, -sgn * bend)
        W2 = (E[0] + f[0] * L2, E[1] + f[1] * L2)
        hand_rot = sgn * out - sgn * bend
        return A, E, W2, hand_rot

    def draw_arm(self, layer, side, out, bend):
        A, E, W2, hr = self._arm_points(side, out, bend)
        g = self.c["arms"][side]; W = self.P(g["W"])
        # smooth curve through the elbow (quadratic Bezier whose control point makes it pass through E)
        C = (2 * E[0] - (A[0] + W2[0]) / 2, 2 * E[1] - (A[1] + W2[1]) / 2)
        pts = []
        for i in range(25):
            t = i / 24
            pts.append(((1 - t) ** 2 * A[0] + 2 * (1 - t) * t * C[0] + t * t * W2[0],
                        (1 - t) ** 2 * A[1] + 2 * (1 - t) * t * C[1] + t * t * W2[1]))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = int(min(xs) - 20), int(min(ys) - 20); x1, y1 = int(max(xs) + 20), int(max(ys) + 20)
        loc = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(loc)
        sp = [((x - x0) * SS, (y - y0) * SS) for x, y in pts]
        sw = self.c["stroke"] * SS
        # root: same thickness as the shirt outline it starts on, so nothing bulges past the shoulder contour
        rw = self.c["outline"] * SS
        d.ellipse((sp[0][0] - rw / 2, sp[0][1] - rw / 2, sp[0][0] + rw / 2, sp[0][1] + rw / 2), fill=(0, 0, 0, 255))
        for i in range(6):                                   # widen gradually from outline width to arm width
            wi = rw + (sw - rw) * (i + 1) / 6
            d.line(sp[i:i + 2], fill=(0, 0, 0, 255), width=int(round(wi)))
            d.ellipse((sp[i + 1][0] - wi / 2, sp[i + 1][1] - wi / 2, sp[i + 1][0] + wi / 2, sp[i + 1][1] + wi / 2), fill=(0, 0, 0, 255))
        d.line(sp[6:], fill=(0, 0, 0, 255), width=sw, joint="curve")
        p = sp[-1]
        d.ellipse((p[0] - sw / 2, p[1] - sw / 2, p[0] + sw / 2, p[1] + sw / 2), fill=(0, 0, 0, 255))
        layer.alpha_composite(loc.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        # hand: rotate about the original wrist, then move the wrist to its new place
        hand = self.hands[side]
        if abs(hr) > 0.05:
            hand = hand.rotate(hr, resample=Image.BILINEAR, center=W)   # bilinear: no ringing specks
        layer.alpha_composite(hand, (int(round(W2[0] - W[0])), int(round(W2[1] - W[1]))))

    # ------------------------------------------------------------------ pose
    def pose(self, arms=None, viseme=1, blink=False, brow=0.0, head_dx=0.0, head_dy=0.0, tilt=0.0, sway=0.0):
        """arms: dict side -> (out, bend). head_dx: sideways drift of the head in px, produced by leaning the whole
        upper body (torso + arms + head) about the hips; head_dy / tilt: nod and tilt about the chin. 'sway' is kept
        for API compatibility (the torso already follows the lean). Returns an RGBA canvas (w + 2*PAD, h + TOP)."""
        arms = arms or {"L": (0, 0), "R": (0, 0)}
        upper = self.body.copy()
        for side in ("L", "R"):
            self.draw_arm(upper, side, *arms[side])
        h = self.face(viseme, blink, brow)
        if abs(tilt) > 0.05: h = h.rotate(tilt, resample=Image.BILINEAR, center=self.head_pivot)
        upper.alpha_composite(h, (0, int(round(head_dy))))
        lean = -math.degrees(math.atan2(head_dx, self.c["lean_radius"]))
        if abs(lean) > 0.02:
            upper = upper.rotate(lean, resample=Image.BILINEAR, center=self.hip_pivot)
        out = self.legs.copy()
        out.alpha_composite(upper)
        return out

# pose library (out, bend) per side, mirrored automatically
POSES = {
    "rest":       {"L": (0, 0),    "R": (0, 0)},
    "tight":      {"L": (-8, 4),   "R": (-8, 4)},          # arms closer to the body (talking default)
    "open":       {"L": (55, 25),  "R": (55, 25)},         # both arms open / 'check this out'
    "hips":       {"L": (35, 95),  "R": (35, 95)},         # hands on hips
    "shrug":      {"L": (40, -75), "R": (40, -75)},        # elbows out, forearms up, palms up
    "present_L":  {"L": (70, 15),  "R": (-12, 115)},       # left arm out to the side, right forearm across the stomach
    "present_R":  {"L": (-12, 115), "R": (70, 15)},
    "point_L":    {"L": (40, -10), "R": (-6, 4)},          # one arm pointing out/down
    "point_R":    {"L": (-6, 4),   "R": (40, -10)},
    "raise_R":    {"L": (-6, 4),   "R": (35, -100)},       # emphasis: forearm up, hand raised by the head
    "raise_L":    {"L": (35, -100), "R": (-6, 4)},
}
