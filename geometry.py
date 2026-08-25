"""Homography estimation, refinement and composition utilities."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_homography(mkpts0, mkpts1, ransac_thresh=3.0, conf=0.999,
                        max_iters=2000):
    """Estimate the homography mapping image-0 -> image-1 (p1 = H @ p0).

    Returns ``(H, inlier_mask)`` or ``(None, None)`` when degenerate.
    """
    mkpts0 = np.asarray(mkpts0, np.float32)
    mkpts1 = np.asarray(mkpts1, np.float32)
    if len(mkpts0) < 4:
        return None, None

    H, mask = cv2.findHomography(
        mkpts0, mkpts1, cv2.RANSAC, ransac_thresh,
        confidence=conf, maxIters=max_iters)
    if H is None:
        return None, None
    return H.astype(np.float64), mask.ravel().astype(bool)


def homography_inliers(mkpts0, mkpts1, H, thresh=4.0):
    """Count symmetric transfer-error inliers of ``H``."""
    if H is None or len(mkpts0) == 0:
        return 0, np.zeros((len(mkpts0),), bool)
    p0 = np.asarray(mkpts0, np.float64)
    p1 = np.asarray(mkpts1, np.float64)

    pts1_est = _apply_h(p0, H)
    pts0_est = _apply_h(p1, np.linalg.inv(H))
    err = 0.5 * (np.linalg.norm(pts1_est - p1, axis=1)
                 + np.linalg.norm(pts0_est - p0, axis=1))
    mask = err < thresh
    return int(mask.sum()), mask


def _apply_h(pts, H):
    ones = np.ones((len(pts), 1))
    hom = np.concatenate([pts, ones], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def transform_points(points, H):
    """Apply homography ``H`` to a set of (N, 2) points."""
    return _apply_h(np.asarray(points, np.float64), H)


def refine_homography(gray0, gray1, H, mode="ecc", max_iters=50, eps=1e-4):
    """Refine ``H`` photometrically (ECC / inverse-compositional Lucas-Kanade).

    ``H`` maps gray0 -> gray1.  Falls back silently to ``H`` on any failure.
    """
    if H is None:
        return None
    try:
        warp_mode = cv2.MOTION_HOMOGRAPHY
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iters, eps)
        _, H_ref = cv2.findTransformECC(
            gray1, gray0, H.astype(np.float32), warp_mode, criteria,
            inputMask=None, gaussFiltSize=5)
        return H_ref.astype(np.float64)
    except cv2.error:
        return H


def compose_homographies(H_left, H_right):
    """Compose two homographies (apply ``H_right`` first, then ``H_left``)."""
    return H_left @ H_right


def inverse_homography(H):
    return np.linalg.inv(H)


def translation_matrix(tx, ty):
    T = np.eye(3)
    T[0, 2] = tx
    T[1, 2] = ty
    return T


def warp_corners(H, w, h):
    """Return the four image corners transformed by ``H`` (clockwise)."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    return _apply_h(corners, H)
