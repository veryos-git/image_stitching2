"""Fit only an x/y offset to learned correspondences; never fit a warp."""
import cv2
import numpy as np


def reliable_translation(source, target, source_shape, target_shape, threshold=3.):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[valid], target[valid]
    if len(source) < 12:
        return None, 0
    offsets = target - source
    # Deterministic displacement consensus tolerates incorrect neural matches.
    rng = np.random.default_rng(0)
    hypotheses = offsets[rng.choice(len(offsets), min(256, len(offsets)), replace=False)]
    counts = [np.count_nonzero(np.linalg.norm(offsets - t, axis=1) <= threshold) for t in hypotheses]
    offset = hypotheses[np.argmax(counts)]
    for _ in range(3):
        mask = np.linalg.norm(offsets - offset, axis=1) <= threshold
        offset = np.median(offsets[mask], axis=0)
    # Whole-pixel placement avoids any resampling of the original images.
    offset = np.rint(offset).astype(np.int64)
    mask = np.linalg.norm(offsets - offset, axis=1) <= threshold
    count = int(mask.sum())
    if count < 12 or count < .5 * len(source):
        return None, count
    for points, shape in ((source[mask], source_shape), (target[mask], target_shape)):
        area = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
        if area < .01 * shape[0] * shape[1]:
            return None, count
    h, w = source_shape
    th, tw = target_shape
    overlap = np.maximum(0, np.minimum(offset + [w, h], [tw, th]) - np.maximum(offset, 0))
    if float(overlap[0]) * float(overlap[1]) < .03 * min(w*h, tw*th):
        return None, count
    return offset, count
