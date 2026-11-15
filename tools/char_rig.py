"""Cut-out rig for a single full-body character image: head tilt, elbow-bent forearms, lip-sync mouth, blink.

All coordinates are in the cut-out image's pixel space (projects/550i/cut/character2.png, 519x1389).
"""
import math
import numpy as np
from PIL import Image, ImageDraw

PAD = 320  # horizontal padding so raised hands never leave the canvas

CFG = dict(
    head_cut=376, head_overlap=10, head_pivot=(262, 382),
    mouth=(275, 297), mouth_clear=(224, 282, 326, 314), skin_pt=(275, 332),
    eyes=[(228, 212), (322, 212)], lens_r=26, lens_pt=(206, 214),
    # forearm: polygon below the cuff, cuff polygon redrawn on top, pivot at the elbow (hidden by the cuff)
    L=dict(fore=[(0, 657), (124, 657), (124, 668), (126, 668), (126, 790), (127, 790), (127, 860), (126, 860), (126, 935), (0, 935)],
           cuff=[(0, 560), (141, 560), (141, 662), (124, 662), (22, 645), (0, 650)],
           pivot=(72, 628), ext_row=668, ext_top=618, xr=(10, 132)),
    R=dict(fore=[(396, 657), (519, 657), (519, 935), (392, 935), (392, 860), (389, 860), (389, 790), (399, 790), (399, 725), (396, 725)],
           cuff=[(378, 560), (519, 560), (519, 650), (497, 645), (396, 662), (378, 662)],
           pivot=(447, 628), ext_row=668, ext_top=618, xr=(390, 512)),
)

CFG3 = dict(
    head_cut=382, head_overlap=10, head_pivot=(229, 388),
    mouth=(229, 326), mouth_clear=(194, 313, 264, 338), skin_pt=(230, 356),
    eyes=[(181, 247), (273, 247)], eye_r=(34, 38), lens_r=None, lens_pt=None,
    brows=[(164, 162, 206, 190), (250, 162, 296, 190)],
    L=dict(fore=[(0, 614), (112, 614), (112, 700), (110, 700), (110, 780), (105, 780), (105, 860), (100, 860), (100, 900), (0, 900)],
           cuff=[(0, 520), (126, 520), (126, 618), (0, 618)], pivot=(66, 586), ext_row=630, ext_top=590, xr=(18, 112)),
    R=dict(fore=[(336, 614), (458, 614), (458, 900), (360, 900), (360, 860), (349, 860), (349, 780), (345, 780), (345, 700), (336, 700)],
           cuff=[(332, 520), (458, 520), (458, 618), (332, 618)], pivot=(392, 586), ext_row=630, ext_top=590, xr=(342, 442)),
    cuff_rule=lambda r: (r.min(-1) > 46) & (r.max(-1) < 80) & (r.max(-1) - r.min(-1) < 14),
    elbow_r=40, cuff_rebuild=True, torso_outline=(600, 712),
    # chain mode: sleeve rotates at the shoulder, upper body leans at the waist
    waist_cut=712, waist_pivot=(229, 712),
    # procedural arm: U (under cuff) -> E (elbow) -> W (wrist), then the original hand bitmap
    arms={"L": dict(U=(93, 596), E=(84.5, 634), W=(57.5, 752), rU=33.5, rE=32.5, rW=29, wrist_cut=746),
          "R": dict(U=(358, 596), E=(368, 634), W=(398.5, 752), rU=33.5, rE=32.5, rW=29, wrist_cut=746)},
    outline=10, skin=(254, 190, 151), shirt=(37, 37, 37),
    armpit={"L": ((100, 447), (125, 612)), "R": ((358, 447), (333, 612))},   # torso seam: shoulder top -> armpit
    sleeves={"L": ([(30, 450), (100, 445), (122, 590), (0, 590), (0, 540)], (100, 447)),
             "R": ([(428, 450), (358, 445), (336, 590), (458, 590), (458, 540)], (358, 447))},
)

# eyebrow offsets per expression: (dy_left, rot_left, dy_right, rot_right); + rot = counter-clockwise
EXPR = {"neutral": (0, 0, 0, 0), "happy": (-2, 0, -2, 0), "surprised": (-6, 0, -6, 0),
        "serious": (2, -12, 2, 12), "curious": (-6, 6, 1, 0)}

def _capsule(d, a, ra, b, rb, col):
    """filled tapered capsule between circle (a, ra) and circle (b, rb)"""
    d.ellipse((a[0] - ra, a[1] - ra, a[0] + ra, a[1] + ra), fill=col)
    d.ellipse((b[0] - rb, b[1] - rb, b[0] + rb, b[1] + rb), fill=col)
    vx, vy = b[0] - a[0], b[1] - a[1]; n = math.hypot(vx, vy) or 1
    px, py = -vy / n, vx / n
    d.polygon([(a[0] + px * ra, a[1] + py * ra), (b[0] + px * rb, b[1] + py * rb),
               (b[0] - px * rb, b[1] - py * rb), (a[0] - px * ra, a[1] - py * ra)], fill=col)

def _poly_mask(size, poly):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).polygon(poly, fill=255); return m

class Rig:
    def __init__(self, path, cfg=None):
        cfg = cfg or CFG
        self.c = cfg
        src = Image.open(path).convert("RGBA")
        self.w, self.h = src.size
        canvas = Image.new("RGBA", (self.w + 2 * PAD, self.h), (0, 0, 0, 0)); canvas.paste(src, (PAD, 0))
        self.src = canvas
        cw = canvas.size
        sh = lambda poly: [(x + PAD, y) for x, y in poly]
        base = canvas.copy()
        self.parts = {}
        for side in ("L", "R"):
            a = cfg[side]
            fmask = _poly_mask(cw, sh(a["fore"]))
            # keep only the largest solid piece (the arm), not slivers of shirt/pants outline
            from scipy import ndimage
            solid = (np.asarray(canvas)[..., 3] > 40) & (np.asarray(fmask) > 0)
            lab, n = ndimage.label(solid)
            if n:
                big = 1 + int(np.argmax(ndimage.sum(solid, lab, range(1, n + 1))))
                keep = ndimage.binary_dilation(lab == big, iterations=2) & (np.asarray(fmask) > 0)
                fmask = Image.fromarray((keep * 255).astype(np.uint8))
            fore = Image.new("RGBA", cw, (0, 0, 0, 0)); fore.paste(canvas, (0, 0), fmask)
            # extend the forearm upward under the cuff by repeating one clean row
            arr = np.array(fore)
            row = arr[a["ext_row"]].copy()
            x0, x1 = a["xr"][0] + PAD, a["xr"][1] + PAD
            arr[a["ext_top"]:a["ext_row"], x0:x1] = row[x0:x1]
            fore = Image.fromarray(arr)
            cmask = _poly_mask(cw, sh(a["cuff"]))
            from scipy import ndimage
            rgb = np.asarray(canvas)[..., :3].astype(int)
            rule = cfg.get("cuff_rule", lambda r: (r[..., 2] > 150) & (r[..., 0] > 95) & (r[..., 2] - r[..., 0] > 25))
            light = rule(rgb) & (np.asarray(cmask) > 0) & (np.asarray(canvas)[..., 3] > 0)
            if cfg.get("cuff_rebuild"):
                # cuff = its own fabric pixels + a freshly drawn uniform outline ring (no stray outline ends)
                light = ndimage.binary_closing(light, iterations=3)
                light = ndimage.binary_fill_holes(light)
                lab, n = ndimage.label(light)
                if n: light = lab == 1 + int(np.argmax(ndimage.sum(light, lab, range(1, n + 1))))
                ow = cfg["outline"]
                dist = ndimage.distance_transform_edt(~light)
                ring_a = np.clip(ow + 0.5 - dist, 0, 1)
                cuff_arr = np.zeros(np.asarray(canvas).shape, np.uint8)
                cuff_arr[light] = np.asarray(canvas)[light]
                ring = (~light) & (ring_a > 0)
                cuff_arr[ring, :3] = 0
                cuff_arr[ring, 3] = (ring_a[ring] * 255).astype(np.uint8)
                band = light | ring
                cuff_img = Image.fromarray(cuff_arr)
            else:
                band = ndimage.binary_dilation(light, iterations=9) & (np.asarray(cmask) > 0)
                cuff_img = None
            cmask = Image.fromarray((band * 255).astype(np.uint8))
            if cuff_img is not None: cuff = cuff_img
            else:
                cuff = Image.new("RGBA", cw, (0, 0, 0, 0)); cuff.paste(canvas, (0, 0), cmask)
            # erase the original forearm from the base, plus the strip just under the cuff (the cuff is redrawn on top)
            b = np.array(base); m = np.array(fmask) > 0
            strip = np.zeros_like(m); xs0, xs1 = a["xr"][0] + PAD - 4, a["xr"][1] + PAD + 2
            strip[a["ext_top"]:a["ext_row"] + 12, xs0:xs1] = True
            m |= strip & ~(np.asarray(cmask) > 0)
            b[m] = 0; base = Image.fromarray(b)
            px, py = a["pivot"]
            # rounded elbow cap drawn under the forearm so the joint never shows a square edge
            r = cfg.get("elbow_r", 44)
            ex = px + PAD
            skin = src.getpixel((ex - PAD, a["ext_row"] + 10))[:3]
            elbow = Image.new("RGBA", cw, (0, 0, 0, 0))
            ImageDraw.Draw(elbow).ellipse((ex - r, py - r, ex + r, py + r), fill=skin + (255,), outline=(0, 0, 0, 255), width=7)
            self.parts[side] = (fore, cuff, (px + PAD, py), elbow)
            self._cmask = getattr(self, "_cmask", {}); self._cmask[side] = np.asarray(cmask) > 0
        self.hands = {}
        if "arms" in cfg:
            for side, g in cfg["arms"].items():
                fore, _, _, _ = self.parts[side]
                arr = np.array(fore); wc = g["wrist_cut"]
                arr[:wc] = 0
                # feather the wrist edge so the hand blends into the forearm fill (no seam line)
                ramp = np.linspace(0, 1, 9)
                for k, v in enumerate(ramp):
                    row = arr[wc + k]; row[..., 3] = (row[..., 3] * v).astype(np.uint8)
                self.hands[side] = Image.fromarray(arr)
        self.chain = "waist_cut" in cfg
        if self.chain:
            b = np.array(base); self.sleeves = {}
            for side, (poly, piv) in cfg["sleeves"].items():
                sm = (np.asarray(_poly_mask(cw, sh(poly))) > 0) & ~self._cmask[side] & (np.asarray(canvas)[..., 3] > 0)
                from scipy import ndimage
                lab, n = ndimage.label(sm)
                if n: sm = lab == 1 + int(np.argmax(ndimage.sum(sm, lab, range(1, n + 1))))
                sl = np.zeros_like(b); src_arr = np.asarray(canvas)
                sl[sm] = src_arr[sm]
                self.sleeves[side] = (Image.fromarray(sl), (piv[0] + PAD, piv[1]))
                # under the moving sleeve/cuff the body shows as black (blends with the outline)
                under = (sm | self._cmask[side]) & (b[..., 3] > 0)
                b[under] = 0   # nothing behind the sleeve: the torso keeps its own drawn seam/outline
            # restore the torso's own side outline where it used to be shared with the arm
            if cfg.get("torso_outline"):
                from scipy import ndimage
                y0, y1 = cfg["torso_outline"]
                orig_a = np.asarray(canvas)[..., 3] > 0
                erased = orig_a & (b[..., 3] == 0)
                erased[:y0] = False; erased[y1:] = False
                near = ndimage.distance_transform_edt(~erased)
                ring = (b[..., 3] > 0) & (near <= cfg["outline"]) & ~erased
                ring[:y0] = False; ring[y1:] = False
                b[ring, :3] = 0; b[ring, 3] = 255
            base = Image.fromarray(b)
        # head
        hc, ho = cfg["head_cut"], cfg["head_overlap"]
        b = np.array(base); head_src = b.copy()
        b[:hc] = 0
        head_src[hc + ho:] = 0
        self.base = Image.fromarray(b)
        if self.chain:
            wc = cfg["waist_cut"]
            up = b.copy(); up[wc + 6:] = 0; lo = b.copy(); lo[:wc - 4] = 0
            self.upper_base, self.lower_base = Image.fromarray(up), Image.fromarray(lo)
            self.waist_pivot = (cfg["waist_pivot"][0] + PAD, cfg["waist_pivot"][1])
        self.head_base = Image.fromarray(head_src)
        self.head_pivot = (cfg["head_pivot"][0] + PAD, cfg["head_pivot"][1])
        self.skin = src.getpixel(cfg["skin_pt"])[:3]
        self.lens = src.getpixel(cfg["lens_pt"])[:3] if cfg.get("lens_pt") else None
        self._heads = {}

    def head(self, level, blink, expr="neutral"):
        k = (level, blink, expr)
        if k in self._heads: return self._heads[k]
        im = self.head_base.copy(); d = ImageDraw.Draw(im); c = self.c
        cx, cy = c["mouth"][0] + PAD, c["mouth"][1]
        if c.get("brows"):
            arr = np.array(im)
            sprites = []
            for (x0, y0, x1, y1) in c["brows"]:
                box = (x0 + PAD, y0, x1 + PAD, y1)
                reg = arr[y0:y1, box[0]:box[2]]
                dark = (reg[..., :3].max(-1) < 90) & (reg[..., 3] > 0)
                spr = np.zeros_like(reg); spr[dark] = reg[dark]
                sprites.append((box, Image.fromarray(spr)))
                reg[dark] = list(self.skin) + [255]
            im = Image.fromarray(arr); d = ImageDraw.Draw(im)
            dyl, rl, dyr, rr = EXPR.get(expr, EXPR["neutral"])
            for (box, spr), dy, rot in zip(sprites, (dyl, dyr), (rl, rr)):
                spr = spr.rotate(rot, resample=Image.BICUBIC, expand=False)
                im.alpha_composite(spr, (box[0], box[1] + dy))
            d = ImageDraw.Draw(im)
        if not level and expr in ("surprised", "serious"):
            x0, y0, x1, y1 = c["mouth_clear"]
            d.ellipse((x0 + PAD, y0, x1 + PAD, y1), fill=self.skin)
            if expr == "surprised":
                d.ellipse((cx - 11, cy - 10, cx + 11, cy + 14), fill=(96, 22, 26), outline=(0, 0, 0), width=4)
            else:
                d.line((cx - 22, cy + 2, cx + 22, cy + 2), fill=(0, 0, 0), width=5)
        if level:
            x0, y0, x1, y1 = c["mouth_clear"]
            d.ellipse((x0 + PAD, y0, x1 + PAD, y1), fill=self.skin)
            hw, hh = [0, 22, 27, 31][level], [0, 9, 15, 22][level]
            box = (cx - hw, cy - 8, cx + hw, cy - 8 + 2 * hh)
            d.ellipse(box, fill=(96, 22, 26))
            mask = Image.new("L", im.size, 0); ImageDraw.Draw(mask).ellipse(box, fill=255)
            tongue = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(tongue).ellipse((cx - hw * 0.6, box[3] - hh * 0.9, cx + hw * 0.6, box[3] + hh * 0.3), fill=(232, 98, 110, 255))
            tongue.putalpha(Image.fromarray(np.minimum(np.asarray(tongue)[..., 3], np.asarray(mask))))
            im.alpha_composite(tongue)
            if level >= 2:
                d.chord((cx - hw + 5, cy - 10, cx + hw - 5, cy + hh * 0.45), 180, 360, fill=(250, 250, 250))
            d.ellipse(box, outline=(0, 0, 0), width=4)
        if blink:
            for ex, ey in c["eyes"]:
                ex += PAD
                if c.get("lens_r"):
                    r = c["lens_r"]; d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=self.lens)
                    d.arc((ex - r + 6, ey - 12, ex + r - 6, ey + 10), 15, 165, fill=(0, 0, 0), width=5)
                else:
                    rx, ry = c["eye_r"]
                    d.ellipse((ex - rx - 2, ey - ry - 2, ex + rx + 2, ey + ry + 2), fill=self.skin)
                    d.arc((ex - rx, ey - 16, ex + rx, ey + 14), 10, 170, fill=(0, 0, 0), width=6)
        self._heads[k] = im
        return im

    def pose(self, left=0.0, right=0.0, tilt=0.0, mouth=0, blink=False, expr="neutral", head_dy=0,
             sh_l=0.0, sh_r=0.0, torso=0.0, breath=0.0):
        if self.chain:
            return self._pose_chain(left, right, tilt, mouth, blink, expr, head_dy, sh_l, sh_r, torso, breath)
        return self._pose_simple(left, right, tilt, mouth, blink, expr, head_dy)

    def _pose_chain(self, left, right, tilt, mouth, blink, expr, head_dy, sh_l, sh_r, torso, breath):
        rot = lambda im, a, c: im.rotate(a, resample=Image.BICUBIC, center=c) if abs(a) > 0.05 else im
        upper = Image.new("RGBA", self.upper_base.size, (0, 0, 0, 0))
        # underarm fabric: fills the opening between the torso seam and a lifted sleeve
        if "armpit" in self.c:
            SS = 2; gd = None
            for side, sh in (("L", sh_l), ("R", -sh_r)):
                (p0, p1) = self.c["armpit"][side]
                P0 = (p0[0] + PAD, p0[1]); P1 = (p1[0] + PAD, p1[1])
                th = math.radians(sh); dx, dy = P1[0] - P0[0], P1[1] - P0[1]
                P1r = (P0[0] + dx * math.cos(th) + dy * math.sin(th), P0[1] - dx * math.sin(th) + dy * math.cos(th))
                outward = (P1r[0] < P1[0]) if side == "L" else (P1r[0] > P1[0])
                th2 = th + math.copysign(math.radians(4), th)   # reach a little under the sleeve: no double-antialiased hairline
                P1u = (P0[0] + dx * math.cos(th2) + dy * math.sin(th2), P0[1] - dx * math.sin(th2) + dy * math.cos(th2))
                # the fabric's lower edge rises from the armpit to a point ~70% down the sleeve's inner edge
                qx, qy = dx * 0.7, dy * 0.7
                Q = (P0[0] + qx * math.cos(th) + qy * math.sin(th), P0[1] - qx * math.sin(th) + qy * math.cos(th))
                Qu = (P0[0] + qx * math.cos(th2) + qy * math.sin(th2), P0[1] - qx * math.sin(th2) + qy * math.cos(th2))
                if not outward or abs(sh) < 0.3: continue
                x0, y0 = int(min(P0[0], P1[0], P1r[0], P1u[0]) - 22), int(P0[1] - 12)
                x1, y1 = int(max(P0[0], P1[0], P1r[0], P1u[0]) + 22), int(max(P1[1], P1r[1], P1u[1]) + 12)
                loc = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0)); gd = ImageDraw.Draw(loc)
                Lp = lambda p: ((p[0] - x0) * SS, (p[1] - y0) * SS)
                inw = 10 if side == "L" else -10   # tuck under the torso so the seam never shows a hairline
                gd.polygon([Lp((P0[0] + inw, P0[1])), Lp((P1[0] + inw, P1[1])), Lp(P1), Lp(Q), Lp(Qu), Lp(P0)], fill=tuple(self.c.get("shirt", (37, 37, 37))) + (255,))
                gd.line([Lp(P1), Lp(Q)], fill=(0, 0, 0, 255), width=self.c["outline"] * SS)
                for q in (P1, Q):
                    r_ = self.c["outline"] * SS / 2; c_ = Lp(q)
                    gd.ellipse((c_[0] - r_, c_[1] - r_, c_[0] + r_, c_[1] + r_), fill=(0, 0, 0, 255))
                upper.alpha_composite(loc.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        upper.alpha_composite(self.upper_base)
        for side, ang, sh in (("L", left, sh_l), ("R", -right, -sh_r)):
            fore, cuff, piv, elbow = self.parts[side]
            sleeve, spiv = self.sleeves[side]
            arm = sleeve.copy()
            if self.hands:
                self._draw_arm(arm, side, ang)
            else:
                if abs(ang) > 70: arm.alpha_composite(elbow)
                arm.alpha_composite(rot(fore, ang, piv))
            arm.alpha_composite(cuff)
            upper.alpha_composite(rot(arm, sh, spiv))
        h = self.head(mouth, blink, expr)
        upper.alpha_composite(rot(h, tilt, self.head_pivot), (0, int(round(head_dy))))
        upper = rot(upper, torso, self.waist_pivot)
        out = self.lower_base.copy()
        out.alpha_composite(upper, (0, -int(round(breath))))
        return out

    def _draw_arm(self, layer, side, ang):
        """upper-arm stub + forearm as one outlined shape (all outlines first, then all fills), hand bitmap at the wrist"""
        g = self.c["arms"][side]; ow = self.c["outline"]; SS = 2
        P = lambda p: (p[0] + PAD, p[1])
        U, E, W = P(g["U"]), P(g["E"]), P(g["W"])
        th = math.radians(ang)
        dx, dy = W[0] - E[0], W[1] - E[1]
        W2 = (E[0] + dx * math.cos(th) + dy * math.sin(th), E[1] - dx * math.sin(th) + dy * math.cos(th))
        pts = [U, E, W2]
        x0 = int(min(p[0] for p in pts) - 45); y0 = int(min(p[1] for p in pts) - 45)
        x1 = int(max(p[0] for p in pts) + 45); y1 = int(max(p[1] for p in pts) + 45)
        loc = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(loc)
        L = lambda p: ((p[0] - x0) * SS, (p[1] - y0) * SS)
        segs = [(U, g["rU"], E, g["rE"]), (E, g["rE"], W2, g["rW"])]
        for grow, col in ((0, (0, 0, 0, 255)), (-ow, tuple(self.c["skin"]) + (255,))):
            for a, ra, b, rb in segs:
                _capsule(d, L(a), (ra + grow) * SS, L(b), (rb + grow) * SS, col)
        loc = loc.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        layer.alpha_composite(loc, (x0, y0))
        hand = self.hands[side]
        layer.alpha_composite(hand.rotate(ang, resample=Image.BICUBIC, center=E) if abs(ang) > 0.05 else hand)

    def _pose_simple(self, left=0.0, right=0.0, tilt=0.0, mouth=0, blink=False, expr="neutral", head_dy=0):
        """left/right: forearm angles in degrees (+ = hand moves toward the body's centre line for both arms)."""
        out = self.base.copy()
        for side, ang in (("L", left), ("R", -right)):
            fore, cuff, piv, elbow = self.parts[side]
            if abs(ang) > 70: out.alpha_composite(elbow)
            out.alpha_composite(fore.rotate(ang, resample=Image.BICUBIC, center=piv) if abs(ang) > 0.05 else fore)
            out.alpha_composite(cuff)
        h = self.head(mouth, blink, expr)
        h = h.rotate(tilt, resample=Image.BICUBIC, center=self.head_pivot) if abs(tilt) > 0.05 else h
        out.alpha_composite(h, (0, int(round(head_dy))))
        return out

GESTURES = {  # (left, right) forearm angles; + = toward the body centre
    "rest": (0, 0), "open": (-30, -30), "open_small": (-16, -16), "out_r": (-4, -42), "out_l": (-42, -4),
    "point_up": (0, 165),
}
