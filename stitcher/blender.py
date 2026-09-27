"""Compositing: canvas layout, warping, exposure compensation, multi-band
blending and automatic border cropping."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import warp_corners, translation_matrix


# --------------------------------------------------------------------------- #
# Canvas layout
# --------------------------------------------------------------------------- #
def compute_canvas(homographies, sizes, margin=20, max_output_dim=12000):
    """Compute the output canvas for a set of global homographies.

    ``homographies[i]`` maps image ``i`` into the reference frame.  Returns
    ``(offset, W, H, scale)``; the final transform of image ``i`` is
    ``translation(offset) @ H_i`` (post-multiplied by ``scale`` if needed).
    """
    all_corners = []
    for H, (h, w) in zip(homographies, sizes):
        all_corners.append(warp_corners(H, w, h))
    all_corners = np.concatenate(all_corners, axis=0)

    xmin = int(np.floor(all_corners[:, 0].min())) - margin
    xmax = int(np.ceil(all_corners[:, 0].max())) + margin
    ymin = int(np.floor(all_corners[:, 1].min())) - margin
    ymax = int(np.ceil(all_corners[:, 1].max())) + margin

    offset = (-xmin, -ymin)
    W = xmax - xmin
    H = ymax - ymin

    # Guard against absurdly large canvases (bounds memory / compute).
    scale = 1.0
    big = max(W, H)
    if big > max_output_dim:
        scale = max_output_dim / float(big)
        W = max(1, int(round(W * scale)))
        H = max(1, int(round(H * scale)))
        offset = (offset[0] * scale, offset[1] * scale)
    return offset, W, H, scale


def final_transform(H, offset, scale=1.0):
    """Full image -> canvas transform (includes the scale guard)."""
    T = translation_matrix(offset[0], offset[1])
    M = T @ H
    if scale != 1.0:
        S = np.diag([scale, scale, 1.0])
        M = S @ M
    return M


def warp_image(img, M, canvas_w, canvas_h):
    """Warp ``img`` (BGR uint8) to the canvas; returns (warped, mask)."""
    warped = cv2.warpPerspective(
        img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full(img.shape[:2], 255, np.uint8), M, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)
    return warped, mask


# --------------------------------------------------------------------------- #
# Feather weights
# --------------------------------------------------------------------------- #
def feather_mask(mask, power=1.0):
    """Distance-transform feather weight for a 0/255 (or boolean) valid mask.

    Returns a float32 HxW array that is 1 in the interior of the valid region
    and ramps smoothly to 0 at its border (raised to ``power``).  This is the
    per-tile weight used by the incremental stitcher's feather blend so that a
    new frame fades in without a hard seam.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return np.zeros(mask.shape[:2], np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    mx = float(dist.max())
    if mx <= 0:
        return m.astype(np.float32)
    wgt = (dist / mx).astype(np.float32)
    if power != 1.0:
        wgt = np.power(wgt, power)
    return np.maximum(wgt, 1e-4) * m


def feather_blend(warped_images, masks, gains=None, power=1.0, progress=None):
    """Normalize distance-to-edge weights, excluding uncovered pixels entirely.

    Pad masks so images touching the canvas edge feather just like interior
    images. Keep coverage separate: even genuinely black pixels are valid.
    """
    if gains is None:
        gains = np.ones(len(warped_images), np.float32)
    total = np.zeros(masks[0].shape, np.float32)
    accum = np.zeros(warped_images[0].shape, np.float32)
    for i, (img, mask, gain) in enumerate(zip(warped_images, masks, gains)):
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        weight = feather_mask(padded, power)[1:-1, 1:-1]
        accum += img.astype(np.float32) * (weight * gain)[..., None]
        total += weight
        if progress:
            progress((i + 1) / len(masks), f"Feathered image {i + 1}/{len(masks)}")
    accum /= np.maximum(total, 1e-8)[..., None]
    return np.clip(np.rint(accum), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Exposure compensation
# --------------------------------------------------------------------------- #
def exposure_compensation(warped_images, masks, max_comp=0.6, progress=None):
    """Estimate per-image exposure gains so overlaps match in brightness."""
    n = len(warped_images)
    A, b = [], []
    bounds = [cv2.boundingRect(mask) for mask in masks]
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            if progress:
                progress(done / max(1, total),
                         f"Exposure overlap {done + 1}/{total} (images {i + 1}, {j + 1})")
            done += 1
            xi, yi, wi, hi = bounds[i]
            xj, yj, wj, hj = bounds[j]
            x0, y0 = max(xi, xj), max(yi, yj)
            x1, y1 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = np.s_[y0:y1, x0:x1]
            inter = (masks[i][roi] > 0) & (masks[j][roi] > 0)
            if inter.sum() < 500:
                continue
            mi = np.median(warped_images[i][roi][inter])
            mj = np.median(warped_images[j][roi][inter])
            if mi < 1.0 or mj < 1.0:
                continue
            # want g_i*mean_i == g_j*mean_j  =>  log g_i - log g_j = log mj - log mi
            row = np.zeros(n)
            row[i], row[j] = 1.0, -1.0
            A.append(row)
            b.append(np.log(mj) - np.log(mi))

    if progress:
        progress(1.0, "Exposure overlaps complete")
    if not A:
        return np.ones(n, np.float32)

    A = np.vstack([np.asarray(A, np.float64),
                   np.ones((1, n)) * 1e-3])  # tiny anchor -> min-norm solution
    b = np.append(np.asarray(b, np.float64), 0.0)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    gains = np.exp(x)
    gains = np.clip(gains, 1.0 - max_comp, 1.0 + max_comp)
    return gains.astype(np.float32)


# --------------------------------------------------------------------------- #
# Multi-band (Laplacian pyramid) blending
# --------------------------------------------------------------------------- #
def _gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr


def _laplacian_pyramid(img, levels):
    gp = _gaussian_pyramid(img, levels)
    lp = [gp[-1]]
    for i in range(levels, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        up = cv2.pyrUp(gp[i], dstsize=size)
        lp.append(cv2.subtract(gp[i - 1], up))
    return lp[::-1]


def multi_band_blend(warped_images, masks, gains=None, levels=6, progress=None):
    """Blend N canvas-aligned images using a Laplacian pyramid."""
    n = len(warped_images)
    if n == 1:
        return np.clip(warped_images[0], 0, 255).astype(np.uint8)

    h, w = masks[0].shape[:2]
    levels = max(1, min(levels, int(np.log2(min(h, w))) - 2))
    if gains is None:
        gains = np.ones(n, np.float32)

    # Accumulate each image immediately instead of retaining N full pyramids.
    blend_pyr, denominators = [], []
    for i in range(n):
        if progress:
            progress(0.8 * (i / n), f"Building pyramid for image {i + 1}/{n}")
        weighted = warped_images[i].astype(np.float32)
        weighted *= gains[i]
        mask = masks[i].astype(np.float32) / 255.0
        weighted *= mask[..., None]
        laps = _laplacian_pyramid(weighted, levels)
        del weighted
        for level, lap in enumerate(laps):
            if i == 0:
                blend_pyr.append(lap)
                denominators.append(mask)
            else:
                blend_pyr[level] += lap
                denominators[level] += mask
            if level < levels:
                mask = cv2.pyrDown(mask)
        del laps, lap, mask
        if progress:
            progress(0.8 * ((i + 1) / n), f"Accumulated image {i + 1}/{n}")

    for level, (num, den) in enumerate(zip(blend_pyr, denominators)):
        if progress:
            progress(0.8 + 0.1 * level / (levels + 1),
                     f"Normalizing blend level {level + 1}/{levels + 1}")
        den[den < 1e-6] = 1.0
        num /= den[..., None]
    del denominators, den

    result = blend_pyr[-1]
    for l in range(levels, 0, -1):
        if progress:
            progress(0.9 + 0.09 * (levels - l) / levels,
                     f"Reconstructing blend level {levels - l + 1}/{levels}")
        size = (blend_pyr[l - 1].shape[1], blend_pyr[l - 1].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, blend_pyr[l - 1])

    if progress:
        progress(1.0, "Multi-band blending complete")
    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def _largest_rect(valid):
    """Largest all-True axis-aligned rectangle; returns (area, x, y, w, h)."""
    H, W = valid.shape
    heights = np.zeros(W, np.int32)
    best = (0, 0, 0, 0, 0)
    stack_x = np.zeros(W, np.int32)
    stack_h = np.zeros(W, np.int32)
    for y in range(H):
        for x in range(W):
            heights[x] = heights[x] + 1 if valid[y, x] else 0
        top = -1
        for x in range(W + 1):
            cur_h = heights[x] if x < W else 0
            start = x
            while top >= 0 and stack_h[top] > cur_h:
                area = stack_h[top] * (x - stack_x[top])
                if area > best[0]:
                    best = (area, stack_x[top], y - stack_h[top] + 1,
                            x - stack_x[top], stack_h[top])
                start = stack_x[top]
                top -= 1
            if top < 0 or cur_h > stack_h[top]:
                top += 1
                stack_x[top] = start
                stack_h[top] = cur_h
    return best


def _exact_trim(weight_mask, bbox, min_weight):
    """Shrink ``bbox`` until its border rows/cols contain no blank pixels."""
    x, y, rw, rh = bbox
    sub = weight_mask[y:y + rh, x:x + rw] > min_weight
    if not sub.any():
        return bbox
    rows = sub.any(axis=1)
    cols = sub.any(axis=0)
    y0 = int(np.argmax(rows))
    y1 = rh - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = rw - int(np.argmax(cols[::-1]))
    return (x + x0, y + y0, x1 - x0, y1 - y0)


def find_crop_bbox(weight_mask, min_weight=0.5, margin=0):
    """Largest fully-valid rectangle inside the accumulated weight mask.

    ``weight_mask``: HxW float array where a pixel's value is the number of
    source images covering it (``>= 1`` means "has content").  The largest
    axis-aligned rectangle is found on a downscaled copy (for speed), scaled
    back up, then trimmed exactly so no black border remains.  ``margin`` is an
    optional *inset* applied to the final rectangle.
    """
    h, w = weight_mask.shape
    small_dim = 1024
    scale = min(1.0, small_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(weight_mask.astype(np.float32),
                           (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = weight_mask
        scale = 1.0

    valid = small > min_weight
    if not valid.any():
        return (0, 0, w, h)

    area, x, y, rw, rh = _largest_rect(valid)
    if area <= 0:
        return (0, 0, w, h)

    x = int(round(x / scale))
    y = int(round(y / scale))
    rw = int(round(rw / scale))
    rh = int(round(rh / scale))

    # Exact pass in full resolution to guarantee no black border survives.
    x, y, rw, rh = _exact_trim(weight_mask, (x, y, rw, rh), min_weight)

    if margin > 0:
        x = min(w, x + margin)
        y = min(h, y + margin)
        rw = max(0, rw - 2 * margin)
        rh = max(0, rh - 2 * margin)

    return (x, y, rw, rh)


def crop(img, bbox):
    x, y, w, h = bbox
    return img[y:y + h, x:x + w]
