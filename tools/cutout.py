"""Remove a plain white background: flood-fill near-white pixels connected to the image border."""
import sys, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

def cutout(src, dst, thr=232, maxdim=1400):
    im = Image.open(src).convert("RGBA")
    a = np.asarray(im).astype(np.int16)
    rgb, alpha0 = a[..., :3], a[..., 3]
    white = (rgb.min(-1) >= thr) & ((rgb.max(-1) - rgb.min(-1)) < 18)
    white |= alpha0 < 10
    lab, _ = ndimage.label(white)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, border[border > 0])
    # soft edge: pixels next to the background get partial alpha by brightness
    fg = ~bg
    edge = fg & ndimage.binary_dilation(bg, iterations=2)
    alpha = np.where(fg, 255, 0).astype(np.float32)
    lum = rgb.mean(-1)
    alpha[edge] = np.clip((255 - lum[edge]) / (255 - thr + 20) * 255, 60, 255)
    alpha = np.asarray(Image.fromarray(alpha.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)))
    alpha = np.minimum(alpha, alpha0)
    out = np.dstack([rgb, alpha]).astype(np.uint8)
    o = Image.fromarray(out, "RGBA")
    o = o.crop(o.getbbox())
    o.thumbnail((maxdim, maxdim), Image.LANCZOS)
    o.save(dst)

if __name__ == "__main__":
    cutout(sys.argv[1], sys.argv[2])
