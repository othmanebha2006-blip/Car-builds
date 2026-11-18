"""Rig for the cap / blue-shirt character (projects/a5/cut/character_v2.png, 507x1302).

Motion model from 11_CHARACTER_MOTION_ANALYSIS_V2.md: static body, head tilts about the neck (behind the shirt collar),
arms snap between poses. Arms are drawn as one outlined limb (upper arm + forearm, real elbow) whose root is hidden
under the sleeve; the original fist drawing is attached at the wrist.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PAD = 330
MOUTH_SCALE = 0.78
SHADE = True
TOP = 120
SS = 2

CFG = dict(
    outline=9, skin=None,
    arms={"L": dict(U=(116, 598), W=(72, 768), r=30), "R": dict(U=(390, 598), W=(435, 768), r=30)},
    wrist_cut=748, sleeve_hem=624,
    neck_pivot=(243, 452), neck_x=(196, 292), collar_y=438,
    mouth_c=(257, 368), mouth_clear=(178, 330, 338, 404),
    eyes=[(193, 270), (312, 270)], eye_r=(36, 45),
)

def _rot(v, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)

def _capsule(d, a, ra, b, rb, col):
    d.ellipse((a[0] - ra, a[1] - ra, a[0] + ra, a[1] + ra), fill=col)
    d.ellipse((b[0] - rb, b[1] - rb, b[0] + rb, b[1] + rb), fill=col)
    vx, vy = b[0] - a[0], b[1] - a[1]; n = math.hypot(vx, vy) or 1
    px, py = -vy / n, vx / n
    d.polygon([(a[0] + px * ra, a[1] + py * ra), (b[0] + px * rb, b[1] + py * rb),
               (b[0] - px * rb, b[1] - py * rb), (a[0] - px * ra, a[1] - py * ra)], fill=col)

class ToonRig:
    def __init__(self, path, cfg=CFG):
        self.c = cfg
        src = Image.open(path).convert("RGBA")
        self.w, self.h = src.size
        cv = Image.new("RGBA", (self.w + 2 * PAD, self.h + TOP), (0, 0, 0, 0)); cv.paste(src, (PAD, TOP))
        A = np.array(cv); size = cv.size
        alpha = A[..., 3] > 0; rgb = A[..., :3].astype(int)
        H, Wd = A.shape[:2]
        Y = np.arange(H)[:, None] * np.ones((1, Wd), int); X = np.ones((H, 1), int) * np.arange(Wd)[None, :]
        self.P = lambda p: (p[0] + PAD, p[1] + TOP)
        dark = (rgb.max(-1) < 90) & alpha
        # skin colour sampled on the forearm
        lx, ly = self.P((86, 700))
        self.skin = tuple(int(v) for v in A[ly, lx, :3])
        # shirt = blue region (+logo) and its outline
        blue = (rgb[..., 2] > 100) & (rgb[..., 2] > rgb[..., 0] + 40) & alpha & (Y > cfg["collar_y"] + TOP - 25)   # shirt + collar (not the cap)
        lab, n = ndimage.label(blue)
        sizes = ndimage.sum(blue, lab, range(1, n + 1))
        shirt = np.isin(lab, 1 + np.flatnonzero(sizes > 300))      # shirt body, sleeves AND the cuff stripes
        shirt = ndimage.binary_fill_holes(ndimage.binary_closing(shirt, iterations=4))
        d_s = ndimage.distance_transform_edt(~shirt)
        shirt_full = shirt | ((d_s <= cfg["outline"] + 2) & dark)
        # arms below the sleeve hem (skin + outline), per side
        hem = cfg["sleeve_hem"] + TOP
        arm_zone = alpha & (Y > hem - 4) & ~shirt_full
        armL = arm_zone & (X < PAD + 128); armR = arm_zone & (X > PAD + 380)
        # keep only the arm components (not the pants)
        def biggest(m):
            lab, n = ndimage.label(m)
            if not n: return m
            return lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
        armL, armR = biggest(armL), biggest(armR)
        # the arm's own corridor (rest pose) below the hem is arm, even where its outline touches the shirt outline
        for side in ("L", "R"):
            g = cfg["arms"][side]; cm = Image.new("L", size, 0)
            _capsule(ImageDraw.Draw(cm), self.P(g["U"]), g["r"] + 1, self.P(g["W"]), g["r"] + 1, 255)
            corr = (np.asarray(cm) > 0) & alpha & (Y > hem + 2) & (Y < cfg["wrist_cut"] + TOP + 4)
            # the sleeve's real bottom edge per column: lowest blue pixel + outline width
            bl = blue & (Y < hem + 40) & (Y > hem - 120)
            ybot = np.where(bl.any(0), (H - 1) - np.argmax(bl[::-1], 0), -1) + cfg["outline"] + 2
            corr &= Y > ybot[None, :]
            if side == "L": armL |= corr
            else: armR |= corr
        for m in (armL, armR):           # anything of the arm hanging below the sleeve edge is arm, never body
            pass
        self.hands = {}
        wc = cfg["wrist_cut"] + TOP
        for side, m in (("L", armL), ("R", armR)):
            hm = m & (Y >= wc)
            hand = np.zeros_like(A); hand[hm] = A[hm]
            # no feathering: the cut sits on the straight forearm (covered by the drawn arm), fist outline stays solid
            self.hands[side] = Image.fromarray(hand)
        arms = armL | armR
        # head = everything above the collar that isn't shirt, plus the neck down to the collar (extended under it)
        nx0, nx1 = cfg["neck_x"][0] + PAD, cfg["neck_x"][1] + PAD
        head_m = alpha & ~shirt_full & ~arms & ((Y < cfg["collar_y"] + TOP) | ((X >= nx0) & (X <= nx1) & (Y < cfg["collar_y"] + TOP + 30)))
        # the shirt's own edge (outline + its soft anti-aliased fringe) stays with the body, except the neck opening
        near_shirt = d_s <= cfg["outline"] + 7
        neck_cols = (X >= nx0) & (X <= nx1)
        head_m &= ~(near_shirt & ~neck_cols & (Y > cfg["collar_y"] + TOP - 60))
        head = np.zeros_like(A); head[head_m] = A[head_m]
        y_ext = cfg["collar_y"] + TOP - 2
        row = head[y_ext]; xs_n = [x for x in range(nx0, nx1 + 1) if row[x, 3] > 0]
        if xs_n:                                          # neck continues under the collar: flat skin + side outlines
            lx, rx = xs_n[0], xs_n[-1]
            head[y_ext:y_ext + 40, lx:rx + 1] = (0, 0, 0, 255)
            head[y_ext:y_ext + 40, lx + cfg["outline"]:rx - cfg["outline"] + 1] = self.skin + (255,)
        self.neck_box = (nx0 + cfg["outline"], cfg["collar_y"] + TOP - 14, nx1 - cfg["outline"], cfg["collar_y"] + TOP + 45)
        base = np.zeros_like(A); bm = alpha & ~head_m & ~arms; base[bm] = A[bm]
        # sleeves: the only part of the body that may cover the arms
        sl = Image.new("L", size, 0); ds = ImageDraw.Draw(sl)
        for poly in ([(0, 380), (165, 380), (152, 500), (139, 646), (0, 646)],
                     [(507, 380), (342, 380), (355, 500), (368, 646), (507, 646)]):
            ds.polygon([self.P(p) for p in poly], fill=255)
        slz = (np.asarray(sl) > 0) & (Y > hem - 60)
        stub = bm & slz & ~shirt_full                      # leftover arm skin/outline under the cuff -> not body
        base[stub] = 0; bm = bm & ~stub
        self.base = Image.fromarray(base)
        self.sleeves = bm & (np.asarray(sl) > 0)          # sleeve shape only (up to the sleeve/torso seam)
        # everything above the sleeve's bottom edge, outside the torso: an arm can't appear there
        bl2 = shirt_full & ((X < PAD + 160) | (X > PAD + 347)) & (Y > hem - 140)
        ybot2 = np.where(bl2.any(0), (H - 1) - np.argmax(bl2[::-1], 0), -1)
        self.above_hem = (Y < ybot2[None, :] - 2) & ((X < PAD + 160) | (X > PAD + 347))
        self.base = Image.fromarray(base)
        self.masks = dict(shirt=shirt_full, arms=arms, head=head_m, alpha=alpha)
        # face: clear the mouth area for visemes, keep the original grin as the rest mouth
        x0, y0, x1, y1 = cfg["mouth_clear"]
        X0, Y0, X1, Y1 = x0 + PAD, y0 + TOP, x1 + PAD, y1 + TOP
        self.grin = Image.fromarray(head[Y0:Y1, X0:X1].copy())
        mask = Image.new("L", size, 0); ImageDraw.Draw(mask).ellipse((X0, Y0, X1, Y1), fill=255)
        mk = np.asarray(mask) > 0
        hrgb = head[..., :3].astype(int)
        notskin = (np.abs(hrgb - np.array(self.skin)).max(-1) > 14) & (head[..., 3] > 0)
        self.head_clean = head.copy()
        self.head_clean[mk & ndimage.binary_dilation(notskin, iterations=2) & (head[..., 3] > 0)] = self.skin + (255,)
        # eye masks (white + pupil + outline) for the blink
        self.eye_m = np.zeros(mk.shape, bool)
        for ex, ey in cfg["eyes"]:
            em = Image.new("L", size, 0); rx, ry = cfg["eye_r"]; ex, ey = self.P((ex, ey))
            ImageDraw.Draw(em).ellipse((ex - rx, ey - ry, ex + rx, ey + ry), fill=255)
            self.eye_m |= (np.asarray(em) > 0) & notskin
        self.eye_m = ndimage.binary_dilation(self.eye_m, iterations=4)
        # the grin alone (largest connected drawing inside the mouth ellipse), for the scaled talking grin
        gm = mk & notskin
        lab, n = ndimage.label(gm)
        if n: gm = lab == 1 + int(np.argmax(ndimage.sum(gm, lab, range(1, n + 1))))
        gm = ndimage.binary_dilation(gm, iterations=1) & (head[..., 3] > 0)
        spr = np.zeros_like(head); spr[gm] = head[gm]
        ys, xs = np.nonzero(gm)
        self.grin_sprite = Image.fromarray(spr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        self.grin_center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
        self.head_full = Image.fromarray(head)
        self.head_clean = Image.fromarray(self.head_clean)
        self.pivot = self.P(cfg["neck_pivot"])
        self._cache = {}

    # -------------------------------------------------------------- face
    def _mouth(self, v):
        """viseme patch (RGBA, local 240x140, centred on mouth_c)"""
        cx, cy = 120 * SS, 60 * SS; K = (0, 0, 0, 255); W_ = (255, 255, 255, 255)
        im = Image.new("RGBA", (240 * SS, 140 * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        hw = 78 * SS; lw = 9 * SS
        def lens(top, bot, n=30):
            up = [(cx - hw + 2 * hw * i / n, cy + top * (1 - ((2 * i / n) - 1) ** 2) * 0 + top) for i in range(n + 1)]
            lo = [(cx + hw - 2 * hw * i / n, cy + bot * (1 - ((2 * i / n) - 1) ** 2)) for i in range(n + 1)]
            return up, lo
        def dmouth(wid, dep, teeth=True, tongue=True):
            """reference-style open mouth: flat top edge, round bottom ('D' rotated), black inside,
            white top-teeth band, red tongue at the bottom"""
            w2 = wid * SS / 2; dp = dep * SS; top = cy - 6 * SS
            n = 40
            bottom = [(cx + w2 * math.cos(math.pi * i / n), top + dp * math.sin(math.pi * i / n)) for i in range(n + 1)]
            shape = [(cx - w2, top), (cx + w2, top)] + bottom[::-1][::-1]
            shape = [(cx - w2, top)] + [(cx - w2 + 2 * w2 * i / n, top - 3 * SS * math.sin(math.pi * i / n)) for i in range(n + 1)] + bottom
            mask = Image.new("L", im.size, 0); ImageDraw.Draw(mask).polygon(shape, fill=255)
            inner = Image.new("RGBA", im.size, (20, 8, 10, 255))
            di = ImageDraw.Draw(inner)
            if teeth:
                di.rectangle((cx - w2, top - 6 * SS, cx + w2, top + max(7, dep * 0.22) * SS), fill=W_)
            if tongue:
                di.ellipse((cx - w2 * 0.62, top + dp * 0.62, cx + w2 * 0.62, top + dp * 1.25), fill=(214, 34, 40, 255))
            im.paste(inner, (0, 0), mask)
            d.line(shape + [shape[0]], fill=K, width=lw, joint="curve")
        if v == 1:   # closed smile
            pts = [(cx - hw + 2 * hw * i / 30, cy + 18 * SS * (1 - ((2 * i / 30) - 1) ** 2)) for i in range(31)]
            d.line(pts, fill=K, width=lw, joint="curve")
        elif v == 3:  # wide open (A, I, open vowels)
            dmouth(150, 58)
        elif v == 7:  # medium open (O, AW)
            dmouth(118, 44)
        elif v == 4:  # small open (OO, U, W)
            dmouth(84, 32, tongue=False)
        k = MOUTH_SCALE                                   # talking mouth a bit smaller than the drawn grin
        sm = im.resize((int(240 * k), int(140 * k)), Image.LANCZOS)
        out = Image.new("RGBA", (240, 140), (0, 0, 0, 0)); out.alpha_composite(sm, ((240 - sm.width) // 2, int(60 - 60 * k)))
        return out

    def face(self, viseme=2, blink=False):
        key = (viseme, blink)
        if key in self._cache: return self._cache[key]
        if viseme in (2, 5):                       # teeth grin = the original drawing (rest / pauses)
            im = self.head_full.copy()
        elif viseme == 8:                          # talking teeth: the same grin, scaled down
            im = self.head_clean.copy()
            k = MOUTH_SCALE; g = self.grin_sprite
            sg = g.resize((int(g.width * k), int(g.height * k)), Image.LANCZOS)
            cx, cy = self.grin_center
            im.alpha_composite(sg, (int(cx - sg.width / 2), int(cy - sg.height / 2 - 3)))
        else:
            im = self.head_clean.copy()
            cx, cy = self.P(self.c["mouth_c"])
            im.alpha_composite(self._mouth(viseme), (int(cx - 120), int(cy - 60)))
        if blink:
            arr = np.array(im); arr[self.eye_m & (arr[..., 3] > 0)] = self.skin + (255,); im = Image.fromarray(arr)
            d = ImageDraw.Draw(im); rx, ry = self.c["eye_r"]
            for ex, ey in self.c["eyes"]:
                ex, ey = self.P((ex, ey))
                d.arc((ex - rx + 2, ey - 26, ex + rx - 2, ey + 14), 15, 165, fill=(0, 0, 0), width=9)
        self._cache[key] = im
        return im

    # -------------------------------------------------------------- arms
    def _draw_arm(self, layer, side, out, bend, part="all"):
        g = self.c["arms"][side]; r = g["r"]; ow = self.c["outline"]
        U, W = self.P(g["U"]), self.P(g["W"])
        sgn = -1 if side == "L" else 1
        v = (W[0] - U[0], W[1] - U[1]); L = math.hypot(*v); L1 = L * 0.5; L2 = L - L1
        u = _rot((v[0] / L, v[1] / L), sgn * out)
        E = (U[0] + u[0] * L1, U[1] + u[1] * L1)
        f = _rot(u, -sgn * bend)
        W2 = (E[0] + f[0] * L2, E[1] + f[1] * L2)
        hr = sgn * out - sgn * bend
        pts = [U, E, W2]
        x0 = int(min(p[0] for p in pts) - r - 4); y0 = int(min(p[1] for p in pts) - r - 4)
        x1 = int(max(p[0] for p in pts) + r + 4); y1 = int(max(p[1] for p in pts) + r + 4)
        loc = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(loc)
        Lp = lambda p: ((p[0] - x0) * SS, (p[1] - y0) * SS)
        segs = {"all": ((U, E), (E, W2)), "upper": ((U, E),), "lower": ((E, W2),)}[part]
        for grow, col in ((0, (0, 0, 0, 255)), (-ow, self.skin + (255,))):
            for a, b in segs:
                _capsule(d, Lp(a), (r + grow) * SS, Lp(b), (r + grow - 1) * SS, col)
        if part == "lower":     # keep the elbow seamless: fill the joint with skin over the upper arm's end outline
            _capsule(d, Lp(E), (r - ow) * SS, Lp(E), (r - ow) * SS, self.skin + (255,))
        layer.alpha_composite(loc.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        if part == "upper": return
        hand = self.hands[side]
        if abs(hr) > 0.05: hand = hand.rotate(hr, resample=Image.BILINEAR, center=W)
        layer.alpha_composite(hand, (int(round(W2[0] - W[0])), int(round(W2[1] - W[1]))))

    def _seg_mask(self, side, pose, part):
        g = self.c["arms"][side]; r = g["r"] + 1
        U, W = self.P(g["U"]), self.P(g["W"]); out, bend = pose
        sgn = -1 if side == "L" else 1
        v = (W[0] - U[0], W[1] - U[1]); L = math.hypot(*v)
        u = _rot((v[0] / L, v[1] / L), sgn * out); E = (U[0] + u[0] * L / 2, U[1] + u[1] * L / 2)
        f = _rot(u, -sgn * bend); W2 = (E[0] + f[0] * L / 2, E[1] + f[1] * L / 2)
        m = Image.new("L", self.base.size, 0); d = ImageDraw.Draw(m)
        if part == "upper": _capsule(d, U, r, E, r, 255)
        else:
            _capsule(d, E, r, W2, r, 255)
            d.rectangle((W2[0] - 70, W2[1] - 20, W2[0] + 70, W2[1] + 110), fill=255) if False else None
        return np.asarray(m) > 0

    # -------------------------------------------------------------- pose
    def _shade(self, out):
        """soft 3D shading: fill regions darken toward their outline on the side away from a top-left light,
        get a faint highlight on the lit side; small contact shadows under the chin/collar and the cap brim."""
        A = np.array(out); rgb = A[..., :3].astype(np.float32); al = A[..., 3]
        ys, xs = np.nonzero(al > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = rgb[y0:y1, x0:x1]; a_ = al[y0:y1, x0:x1]
        mx = sub.max(-1); mn = sub.min(-1)
        fill = (a_ > 200) & (mx > 70) & ~((mn > 235))          # coloured fills (not black lines, not whites)
        dist = ndimage.distance_transform_edt(fill)
        gy, gx = np.gradient(ndimage.gaussian_filter(dist, 2.0))
        nrm = np.hypot(gx, gy) + 1e-6
        Lx, Ly = -0.55, -0.83                                   # light from the top-left
        facing = (gx * Lx + gy * Ly) / nrm                      # gradient points inward: >0 = edge faces away from light
        rim = np.exp(-dist / 16.0)
        dark = np.clip(facing, 0, 1) * rim * 0.30 + rim * 0.06  # shadow side + slight overall volume
        lite = np.clip(-facing, 0, 1) * np.exp(-dist / 7.0) * 0.10
        nb = self.neck_box; nm = np.zeros(fill.shape, bool)
        nm[max(nb[1] - y0, 0):max(nb[3] - y0, 0), max(nb[0] - x0, 0):max(nb[2] - x0, 0)] = True
        skinpx = np.abs(sub - np.array(self.skin, np.float32)).max(-1) < 40
        nm &= skinpx                                            # only the neck's skin, never the collar
        dark = np.where(nm, 0.0, dark); lite = np.where(nm, 0.0, lite)
        k = (1 - dark + lite)[..., None]
        sub2 = np.where(fill[..., None], np.clip(sub * k, 0, 255), sub)
        # contact shadow under the chin: darkest right below the chin, fading down the neck (no hard edges)
        yy = np.arange(fill.shape[0])[:, None] + y0
        chin_shadow = np.where(nm, 0.22 * np.exp(-np.clip(yy - nb[1], 0, None) / 14.0), 0.0)
        sub2 = np.where(fill[..., None], sub2 * (1 - chin_shadow[..., None]), sub2)
        A[y0:y1, x0:x1, :3] = sub2.astype(np.uint8)
        return Image.fromarray(A)

    def pose(self, arms=None, viseme=2, blink=False, brow=0.0, head_dx=0.0, head_dy=0.0, tilt=0.0, sway=0.0):
        arms = arms or POSES["rest"]
        out = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        h = self.face(viseme, blink)
        ang = tilt - math.degrees(math.atan2(head_dx, 420))       # sideways drift = tilt about the neck
        if abs(ang) > 0.02: h = h.rotate(ang, resample=Image.BILINEAR, center=self.pivot)
        out.alpha_composite(h, (0, int(round(head_dy))))
        for side in ("L", "R"):                                      # arm roots go under the sleeves
            self._draw_arm(out, side, *arms[side])
        out.alpha_composite(self.base)
        # the hands must stay in front of the body: redraw the parts of the arms outside the sleeves on top
        for side in ("L", "R"):
            full = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
            self._draw_arm(full, side, *arms[side])                 # whole arm, one shape: seamless elbow
            fa = np.array(full)
            up = self._seg_mask(side, arms[side], "upper") & ~self._seg_mask(side, arms[side], "lower")
            hide = up & (self.sleeves | self.above_hem)
            fa[hide] = 0                                            # only the upper arm goes under the sleeve
            out.alpha_composite(Image.fromarray(fa))
        return self._shade(out) if SHADE else out

POSES = {   # (out, bend) per side; snap between these
    "rest":      {"L": (0, 0),     "R": (0, 0)},
    "open":      {"L": (32, -26),  "R": (32, -26)},       # arms open wide, confident
    "shrug":     {"L": (28, -85),  "R": (28, -85)},       # both forearms up and out (palms-up shrug)
    "raise_R":   {"L": (0, 0),     "R": (28, -135)},      # fist raised high by the shoulder (emphasis)
    "raise_L":   {"L": (28, -135), "R": (0, 0)},
    "raise_R_hip": {"L": (30, 70), "R": (28, -135)},      # fist up + other hand on the hip (power pose)
    "raise_L_hip": {"L": (28, -135), "R": (30, 70)},
    "out_R":     {"L": (4, 0),     "R": (74, -8)},        # arm pointing straight out to the side
    "out_L":     {"L": (74, -8),   "R": (4, 0)},
    "out_R_hip": {"L": (30, 70),   "R": (74, -8)},        # point out + hand on hip
    "out_L_hip": {"L": (74, -8),   "R": (30, 70)},
    "hips":      {"L": (30, 70),   "R": (30, 70)},        # elbows out, fists at the hips
}
