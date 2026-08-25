"""High-level panorama stitcher with marker-based progress reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from . import blender
from .geometry import (estimate_homography, refine_homography,
                       homography_inliers)
from .matching import build_engine
from .progress import ProgressReporter
from .utils import (load_image, encode_b64, draw_keypoints, draw_matches,
                    resize_to_max_dim)


class StitchError(RuntimeError):
    pass


class NoExtensionError(StitchError):
    """Raised when an image matches but does not extend the map enough.

    Used by the incremental stitcher's ``min_extend_*`` gate to signal a
    soft-skip (the frame is valid but redundant) rather than a hard failure.
    """

    def __init__(self, message, added_area=0.0, required_area=0.0):
        super().__init__(message)
        self.added_area = float(added_area)
        self.required_area = float(required_area)


# Speed / quality presets.  ``feature_max_dim`` is the single biggest speed
# lever: SuperPoint runs on an image downscaled to that many pixels, while the
# final panorama is still rendered at full resolution.  ``max_keypoints`` and
# ``sinkhorn_iterations`` control SuperGlue's attention cost (O(K^2)).
PRESETS = {
    "fast": {
        "feature_max_dim": 512,
        "max_keypoints": 512,
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    },
    "balanced": {
        "feature_max_dim": 1024,
        "max_keypoints": 1024,
        "sinkhorn_iterations": 50,
        "match_threshold": 0.2,
    },
    "best": {
        "feature_max_dim": 1600,
        "max_keypoints": 2048,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    },
}


@dataclass
class StitchConfig:
    # Engine / features
    engine: str = "superglue"          # superglue | sift | orb
    superglue_weights: str = "outdoor"  # indoor | outdoor
    feature_max_dim: int = 1024
    max_keypoints: int = 1024
    keypoint_threshold: float = 0.005
    nms_radius: int = 4
    match_threshold: float = 0.2
    sinkhorn_iterations: int = 50
    ratio_test: float = 0.75           # classic matcher ratio test
    # Geometry / alignment
    ransac_thresh: float = 3.0
    refine: bool = False               # photometric (ECC) refinement
    reference: str = "middle"          # middle | first
    # Compositing
    exposure: bool = True
    blend_levels: int = 6
    crop: bool = True
    crop_margin: int = 0
    max_output_dim: int = 12000
    # Incremental "growing map" gate
    min_extend_ratio: float = 0.0    # min new bounding-box area as fraction of map
    min_extend_px: float = 0.0       # min new bounding-box area in px²
    # Assets / output
    weights_dir: str = "weights"
    viz: bool = True

    def to_dict(self):
        return asdict(self)

    def with_preset(self, preset: str) -> "StitchConfig":
        p = PRESETS.get((preset or "").lower())
        if p:
            for k, v in p.items():
                setattr(self, k, v)
        return self


@dataclass
class StitchResult:
    panorama: np.ndarray
    stats: dict = field(default_factory=dict)


class PanoramaStitcher:
    def __init__(self, config: StitchConfig | None = None,
                 reporter: ProgressReporter | None = None):
        self.config = config or StitchConfig()
        self.reporter = reporter or ProgressReporter()
        self.engine = None

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    def stitch_paths(self, paths):
        bgr, gray = [], []
        for p in paths:
            b, g = load_image(p)
            bgr.append(b)
            gray.append(g)
        return self.stitch(bgr, gray)

    def stitch(self, images, grays=None):
        """Stitch a list of BGR ``images`` (optionally with grayscale copies).

        Returns a :class:`StitchResult`.
        """
        cfg = self.config
        rep = self.reporter
        t0 = time.time()

        rep.plan({
            "engine": cfg.engine,
            "weights": cfg.superglue_weights,
            "num_images": len(images),
            "reference": cfg.reference,
        })

        if len(images) < 2:
            raise StitchError("Need at least 2 images to stitch a panorama.")

        # ---- prepare ---------------------------------------------------- #
        rep.stage_start("prepare", f"Loading {len(images)} images…")
        if grays is None:
            grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        if len(images) != len(grays):
            raise StitchError("images/gray length mismatch")
        sizes = [g.shape[:2] for g in grays]
        rep.stage_done(
            "prepare",
            f"{len(images)} images ready",
            {"sizes": [{"w": int(s[1]), "h": int(s[0])} for s in sizes]},
        )

        # ---- features ---------------------------------------------------- #
        rep.stage_start("features", f"Extracting features with "
                                    f"'{cfg.engine}'…")
        self.engine = build_engine(cfg.to_dict())
        rep.log(f"Engine: {self.engine.name}", "info", "features")

        features = []
        for i, g in enumerate(grays):
            rep.stage_progress("features", (i + 1) / len(images),
                               f"Image {i + 1}/{len(images)}")
            fs = self.engine.extract(g)
            features.append(fs)
            if cfg.viz:
                kpts_small = fs.keypoints * fs.scale
                rep.image(
                    "features",
                    encode_b64(draw_keypoints(fs.small_gray, kpts_small)),
                    f"Keypoints #{i + 1}",
                    f"{len(fs.keypoints)} keypoints",
                    {"image": i + 1, "keypoints": int(len(fs.keypoints))},
                )
        total_kpts = sum(len(f.keypoints) for f in features)
        rep.stage_done(
            "features",
            f"{total_kpts} keypoints total",
            {"per_image": [int(len(f.keypoints)) for f in features],
             "elapsed": round(time.time() - t0, 3)},
        )

        # ---- match ------------------------------------------------------- #
        n = len(features)
        rep.stage_start("match", f"Matching {n - 1} pairs…")
        matches = []
        for i in range(n - 1):
            rep.stage_progress("match", (i + 1) / (n - 1),
                               f"Pair {i + 1}↔{i + 2}")
            mkpts0, mkpts1, conf = self.engine.match(features[i],
                                                     features[i + 1])
            matches.append((mkpts0, mkpts1, conf))
            if cfg.viz:
                m0 = mkpts0 * features[i].scale
                m1 = mkpts1 * features[i + 1].scale
                viz = draw_matches(features[i].small_gray,
                                   features[i + 1].small_gray, m0, m1, conf)
                rep.image("match", encode_b64(viz),
                          f"Matches {i + 1}↔{i + 2}",
                          f"{len(mkpts0)} matches",
                          {"pair": [i + 1, i + 2], "matches": int(len(mkpts0))})
        rep.stage_done("match",
                       " ".join(f"{len(m[0])}" for m in matches) + " matches/pair",
                       {"matches_per_pair": [int(len(m[0])) for m in matches]})

        # ---- align ------------------------------------------------------- #
        rep.stage_start("align", "Estimating homographies…")
        pairwise_H = []
        for i, (mkpts0, mkpts1, _conf) in enumerate(matches):
            rep.stage_progress("align", (i + 1) / (n - 1),
                               f"Homography {i + 1}→{i + 2}")
            if len(mkpts0) < 4:
                raise StitchError(
                    f"Not enough matches ({len(mkpts0)}) between image "
                    f"{i + 1} and {i + 2}. Try a different engine or lower "
                    "the match threshold.")
            H, _ = estimate_homography(mkpts0, mkpts1, cfg.ransac_thresh)
            if H is None:
                raise StitchError(
                    f"Homography between image {i + 1} and {i + 2} is "
                    "degenerate (likely no overlap or repetitive scene).")
            if cfg.refine:
                H = refine_homography(grays[i], grays[i + 1], H)
            inliers, _ = homography_inliers(mkpts0, mkpts1, H)
            pairwise_H.append(H)
            rep.log(f"H {i + 1}→{i + 2}: {inliers}/{len(mkpts0)} inliers",
                    "info", "align")

        ref = 0 if cfg.reference == "first" else n // 2
        Hs = self._global_homographies(pairwise_H, ref)
        rep.stage_done(
            "align",
            f"Reference image #{ref + 1}",
            {"reference": ref + 1,
             "inliers_per_pair": [int(homography_inliers(m[0], m[1], H)[0])
                                  for m, H in zip(matches, pairwise_H)]},
        )

        # ---- blend ------------------------------------------------------- #
        rep.stage_start("blend", "Warping & multi-band blending…")
        rep.stage_progress("blend", 0.2, "Computing canvas")
        offset, W, H, scale = blender.compute_canvas(
            Hs, sizes, max_output_dim=cfg.max_output_dim)

        warped, masks = [], []
        for i, img in enumerate(images):
            M = blender.final_transform(Hs[i], offset, scale)
            w_img, w_mask = blender.warp_image(img, M, W, H)
            warped.append(w_img)
            masks.append(w_mask)
            rep.stage_progress("blend", 0.3 + 0.3 * (i + 1) / n,
                               f"Warped image {i + 1}/{n}")

        gains = np.ones(n, np.float32)
        if cfg.exposure and n > 1:
            rep.stage_progress("blend", 0.65, "Exposure compensation")
            gains = blender.exposure_compensation(warped, masks)

        rep.stage_progress("blend", 0.8,
                           f"Multi-band blending ({cfg.blend_levels} levels)")
        blended = blender.multi_band_blend(warped, masks, gains,
                                           cfg.blend_levels)

        if cfg.viz:
            preview, _ = resize_to_max_dim(blended, 1600)
            rep.image("blend", encode_b64(preview),
                      "Blended (uncropped)", f"{W}×{H} px",
                      {"width": W, "height": H})

        weight_sum = np.zeros((H, W), np.float32)
        for m in masks:
            weight_sum += m.astype(np.float32) / 255.0

        rep.stage_done("blend",
                       f"Canvas {W}×{H} px",
                       {"canvas": {"w": W, "h": H},
                        "gains": [round(float(g), 3) for g in gains],
                        "downscaled": scale != 1.0})

        # ---- crop -------------------------------------------------------- #
        rep.stage_start("crop", "Auto-cropping borders…")
        if cfg.crop:
            bbox = blender.find_crop_bbox(weight_sum, margin=cfg.crop_margin)
            result = blender.crop(blended, bbox)
            rep.stage_done("crop", f"Cropped to {bbox[2]}×{bbox[3]} px",
                           {"bbox": list(bbox)})
        else:
            result = blended
            rep.stage_done("crop", "Skipped", {"bbox": [0, 0, W, H]})

        stats = {
            "engine": self.engine.name,
            "num_images": n,
            "canvas": {"w": W, "h": H},
            "output": {"w": result.shape[1], "h": result.shape[0]},
            "elapsed": round(time.time() - t0, 3),
            "reference": ref + 1,
        }
        rep.result(encode_b64(result, ".jpg", 95), result.shape[1],
                   result.shape[0], stats)
        return StitchResult(result, stats)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _global_homographies(pairwise_H, ref):
        """Chain pairwise (image i -> i+1) homographies into the ref frame."""
        n = len(pairwise_H) + 1
        Hs = [None] * n
        Hs[ref] = np.eye(3)
        for i in range(ref - 1, -1, -1):
            Hs[i] = pairwise_H[i] @ Hs[i + 1]
        for i in range(ref + 1, n):
            Hs[i] = np.linalg.inv(pairwise_H[i - 1]) @ Hs[i - 1]
        return Hs
