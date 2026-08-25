"""Feature extraction + matching engines.

Two families are provided:

* ``SuperGlueEngine`` -- SuperPoint keypoints + SuperGlue graph matching
  (the "AI" path, strongest but heavier);
* ``ClassicEngine``  -- OpenCV SIFT or ORB with a ratio-tested brute-force
  matcher (very fast, CPU friendly fallback).

Both expose an identical interface so the stitcher is agnostic to the engine.
Keypoint coordinates are always returned in the *original* image resolution.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .models import SuperPoint, SuperGlue
from .utils import resize_to_max_dim

# Cache of loaded checkpoint *state dicts*.  The checkpoints (~100 MB total) are
# the slow part of constructing an engine; caching them in memory means repeat
# runs (e.g. consecutive web requests) skip the disk + torch.load cost.  Each
# engine still gets its own model instance (thread-safe, per-request config).
_STATE_CACHE: dict = {}
_STATE_LOCK = threading.Lock()


def _cached_state(path):
    with _STATE_LOCK:
        if path not in _STATE_CACHE:
            _STATE_CACHE[path] = torch.load(path, map_location="cpu",
                                            weights_only=False)
        return _STATE_CACHE[path]


@dataclass
class FeatureSet:
    keypoints: np.ndarray          # [N, 2] float32, original image coords
    descriptors: np.ndarray        # [N, D]
    scores: np.ndarray             # [N]
    shape: tuple                   # (H, W) of the *original* image
    scale: float = 1.0             # small_gray -> original resolution factor
    small_gray: np.ndarray = field(default=None)  # downscaled gray for viz


def _empty_features(shape, desc_dim):
    return FeatureSet(
        keypoints=np.zeros((0, 2), np.float32),
        descriptors=np.zeros((0, desc_dim), np.float32),
        scores=np.zeros((0,), np.float32),
        shape=shape,
    )


class SuperGlueEngine:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        weights_dir = config.get("weights_dir")
        sp_path = os.path.join(weights_dir, "superpoint_v1.pth")
        sg_path = os.path.join(weights_dir,
                               f"superglue_{config['superglue_weights']}.pth")

        sp_cfg = {
            "nms_radius": config.get("nms_radius", 4),
            "keypoint_threshold": config.get("keypoint_threshold", 0.005),
            "max_keypoints": config.get("max_keypoints", 1024),
            "remove_borders": 4,
        }
        sg_cfg = {
            "weights": config.get("superglue_weights", "outdoor"),
            "sinkhorn_iterations": config.get("sinkhorn_iterations", 100),
            "match_threshold": config.get("match_threshold", 0.2),
        }

        # Build fresh model instances from the cached checkpoint state (avoids
        # re-reading ~100 MB of weights from disk on every request).
        self.superpoint = SuperPoint(sp_cfg, weights_path=None)
        self.superpoint.load_state_dict(_cached_state(sp_path))
        self.superpoint.eval().to(device)

        self.superglue = SuperGlue(sg_cfg, weights_path=None)
        self.superglue.load_state_dict(_cached_state(sg_path))
        self.superglue.eval().to(device)
        self.name = f"superglue-{sg_cfg['weights']}"

    def extract(self, gray):
        gray_small, scale = resize_to_max_dim(
            gray, self.config.get("feature_max_dim", 1200))
        h, w = gray_small.shape
        tensor = torch.from_numpy(gray_small.astype(np.float32) / 255.0)[None, None].to(self.device)
        out = self.superpoint(tensor)
        kpts = out["keypoints"].cpu().numpy()          # [N, 2] at small res
        desc = out["descriptors"].cpu().numpy().T      # [N, D]
        scores = out["scores"].cpu().numpy()
        if len(kpts):
            kpts = kpts / scale  # back to original resolution
        return FeatureSet(
            keypoints=kpts.astype(np.float32),
            descriptors=desc.astype(np.float32),
            scores=scores.astype(np.float32),
            shape=gray.shape[:2],
            scale=scale,
            small_gray=gray_small,
        )

    def match(self, fs0, fs1):
        if len(fs0.keypoints) == 0 or len(fs1.keypoints) == 0:
            return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        dev = self.device
        k0 = torch.from_numpy(fs0.keypoints).float().to(dev)
        k1 = torch.from_numpy(fs1.keypoints).float().to(dev)
        d0 = torch.from_numpy(fs0.descriptors).float().to(dev)
        d1 = torch.from_numpy(fs1.descriptors).float().to(dev)
        s0 = torch.from_numpy(fs0.scores).float().to(dev)
        s1 = torch.from_numpy(fs1.scores).float().to(dev)
        shp0 = (1, 1, fs0.shape[0], fs0.shape[1])
        shp1 = (1, 1, fs1.shape[0], fs1.shape[1])

        matches0, mscores0 = self.superglue(k0, k1, d0, d1, s0, s1, shp0, shp1)
        matches0 = matches0.cpu().numpy()
        mscores0 = mscores0.cpu().numpy()

        valid = matches0 > -1
        mkpts0 = fs0.keypoints[valid]
        mkpts1 = fs1.keypoints[matches0[valid]]
        conf = mscores0[valid]
        return mkpts0, mkpts1, conf


class ClassicEngine:
    def __init__(self, config, kind="orb"):
        self.config = config
        self.kind = kind
        nfeat = config.get("max_keypoints", 2048)
        if kind == "orb":
            self.detector = cv2.ORB_create(nfeatures=nfeat)
            self.norm = cv2.NORM_HAMMING
            self.name = "orb"
        else:
            self.detector = cv2.SIFT_create(nfeatures=nfeat)
            self.norm = cv2.NORM_L2
            self.name = "sift"
        self.matcher = cv2.BFMatcher(self.norm)
        self.ratio = config.get("ratio_test", 0.75)

    def extract(self, gray):
        gray_small, scale = resize_to_max_dim(
            gray, self.config.get("feature_max_dim", 1600))
        kpts, desc = self.detector.detectAndCompute(gray_small, None)
        if desc is None or len(kpts) == 0:
            return _empty_features(gray.shape[:2], 0)
        kp_arr = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], np.float32) / scale
        return FeatureSet(
            keypoints=kp_arr,
            descriptors=desc,
            scores=np.ones(len(kpts), np.float32),
            shape=gray.shape[:2],
            scale=scale,
            small_gray=gray_small,
        )

    def match(self, fs0, fs1):
        if len(fs0.keypoints) == 0 or len(fs1.keypoints) == 0:
            return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        raw = self.matcher.knnMatch(fs0.descriptors, fs1.descriptors, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio * n.distance:
                    good.append(m)
            elif len(pair) == 1:
                good.append(pair[0])

        if not good:
            return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        idx0 = np.array([m.queryIdx for m in good])
        idx1 = np.array([m.trainIdx for m in good])
        dist = np.array([m.distance for m in good], np.float32)
        conf = 1.0 / (1.0 + dist)
        return fs0.keypoints[idx0], fs1.keypoints[idx1], conf


def build_engine(config, device=None):
    """Instantiate the requested matching engine."""
    # Inference only.  SuperPoint/SuperGlue use small conv/attention ops, where
    # a handful of threads beats "all cores" (oversubscription thrashes cache).
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    except Exception:  # noqa: BLE001
        pass

    name = (config.get("engine") or "superglue").lower()
    if name in ("superglue", "superpoint+superglue"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return SuperGlueEngine(config, device)
    if name == "orb":
        return ClassicEngine(config, "orb")
    if name == "sift":
        return ClassicEngine(config, "sift")
    raise ValueError(f"Unknown engine: {name}")
