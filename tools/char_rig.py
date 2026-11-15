"""Cut-out rig for a single full-body character image: head tilt, elbow-bent forearms, lip-sync mouth, blink.

All coordinates are in the cut-out image's pixel space (projects/550i/cut/character2.png, 519x1389).
"""
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

def _poly_mask(size, poly):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).polygon(poly, fill=255); return m

class Rig:
    def __init__(self, path, cfg=CFG):
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
            # only the light-blue cuff band (+ its outline), not the sleeve above it
            from scipy import ndimage
            rgb = np.asarray(canvas)[..., :3].astype(int)
            light = (rgb[..., 2] > 150) & (rgb[..., 0] > 95) & (rgb[..., 2] - rgb[..., 0] > 25)
            band = ndimage.binary_dilation(light & (np.asarray(cmask) > 0), iterations=9) & (np.asarray(cmask) > 0)
            cmask = Image.fromarray((band * 255).astype(np.uint8))
            cuff = Image.new("RGBA", cw, (0, 0, 0, 0)); cuff.paste(canvas, (0, 0), cmask)
            # erase the original forearm from the base
            b = np.array(base); m = np.array(fmask) > 0; b[m] = 0; base = Image.fromarray(b)
            px, py = a["pivot"]
            # rounded elbow cap drawn under the forearm so the joint never shows a square edge
            r = 44
            ex = px + PAD
            skin = src.getpixel((ex - PAD, a["ext_row"] + 10))[:3]
            elbow = Image.new("RGBA", cw, (0, 0, 0, 0))
            ImageDraw.Draw(elbow).ellipse((ex - r, py - r, ex + r, py + r), fill=skin + (255,), outline=(0, 0, 0, 255), width=7)
            self.parts[side] = (fore, cuff, (px + PAD, py), elbow)
        # head
        hc, ho = cfg["head_cut"], cfg["head_overlap"]
        b = np.array(base); head_src = b.copy()
        b[:hc] = 0
        head_src[hc + ho:] = 0
        self.base = Image.fromarray(b)
        self.head_base = Image.fromarray(head_src)
        self.head_pivot = (cfg["head_pivot"][0] + PAD, cfg["head_pivot"][1])
        self.skin = src.getpixel(cfg["skin_pt"])[:3]
        self.lens = src.getpixel(cfg["lens_pt"])[:3]
        self._heads = {}

    def head(self, level, blink):
        k = (level, blink)
        if k in self._heads: return self._heads[k]
        im = self.head_base.copy(); d = ImageDraw.Draw(im); c = self.c
        cx, cy = c["mouth"][0] + PAD, c["mouth"][1]
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
            r = c["lens_r"]
            for ex, ey in c["eyes"]:
                ex += PAD
                d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=self.lens)
                d.arc((ex - r + 6, ey - 12, ex + r - 6, ey + 10), 15, 165, fill=(0, 0, 0), width=5)
        self._heads[k] = im
        return im

    def pose(self, left=0.0, right=0.0, tilt=0.0, mouth=0, blink=False):
        """left/right: forearm angles in degrees (+ = hand moves toward the body's centre line for both arms)."""
        out = self.base.copy()
        for side, ang in (("L", left), ("R", -right)):
            fore, cuff, piv, elbow = self.parts[side]
            if abs(ang) > 70: out.alpha_composite(elbow)
            out.alpha_composite(fore.rotate(ang, resample=Image.BICUBIC, center=piv) if abs(ang) > 0.05 else fore)
            out.alpha_composite(cuff)
        h = self.head(mouth, blink)
        out.alpha_composite(h.rotate(tilt, resample=Image.BICUBIC, center=self.head_pivot) if abs(tilt) > 0.05 else h)
        return out

GESTURES = {  # (left, right) forearm angles; + = toward the body centre
    "rest": (0, 0), "open": (-30, -30), "open_small": (-16, -16), "out_r": (-4, -42), "out_l": (-42, -4),
    "point_up": (0, 165),
}
