"""Small image / encoding helpers shared by the core and the web layer."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path):
    """Load an image, preserving colour and EXIF orientation.

    Returns ``(bgr, gray, original_size)``.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def resize_to_max_dim(img, max_dim):
    """Resize so the largest side is ``max_dim``; returns resized image + scale."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def encode_b64(img, ext=".jpg", quality=90):
    """Encode a BGR/RGB/gray numpy image to a ``data:`` URL string."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality]
                           if ext in (".jpg", ".jpeg") else [])
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")
    return f"data:image/{mime};base64,{b64}"


def gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_keypoints(gray, kpts, color=(0, 255, 0), radius=3, max_draw=4000):
    """Overlay keypoints on a grayscale image (returns BGR)."""
    out = gray_to_bgr(gray)
    kpts = np.asarray(kpts, dtype=np.float32)
    step = max(1, len(kpts) // max_draw)
    for x, y in kpts[::step]:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, 1,
                   lineType=cv2.LINE_AA)
    return out


def draw_matches(gray0, gray1, mkpts0, mkpts1, conf=None, max_draw=2000):
    """Side-by-side match visualisation (returns BGR)."""
    H0, W0 = gray0.shape[:2]
    H1, W1 = gray1.shape[:2]
    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = gray_to_bgr(gray0)
    out[:H1, W0:] = gray_to_bgr(gray1)

    mkpts0 = np.asarray(mkpts0, dtype=np.float32)
    mkpts1 = np.asarray(mkpts1, dtype=np.float32)
    if conf is None:
        conf = np.ones(len(mkpts0))
    conf = np.asarray(conf, dtype=np.float32)
    cmin, cmax = conf.min(), conf.max()
    if cmax - cmin < 1e-6:
        norm = np.ones_like(conf)
    else:
        norm = (conf - cmin) / (cmax - cmin)

    idx = np.arange(len(mkpts0))
    if len(idx) > max_draw:
        idx = np.random.default_rng(0).choice(idx, max_draw, replace=False)

    for i in idx:
        c = int(255 * norm[i])
        # red -> green by confidence (low -> high)
        color = (0, c, 255 - c)
        x0, y0 = int(mkpts0[i, 0]), int(mkpts0[i, 1])
        x1, y1 = int(mkpts1[i, 0]) + W0, int(mkpts1[i, 1])
        cv2.line(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 3, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 3, color, -1, cv2.LINE_AA)
    return out


def discover_images(directory):
    """Return sorted image files under ``directory``."""
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*")
                   if p.suffix.lower() in SUPPORTED_EXTS)
    return [str(p) for p in files]
