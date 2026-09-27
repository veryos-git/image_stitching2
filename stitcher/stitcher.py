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
    engine_options: dict = field(default_factory=dict)
    diagnostics_dir: str | None = None
    diagnostics_url: str = ""
    # Geometry / alignment
    alignment: str = "homography"      # homography | translation
    ransac_thresh: float = 3.0
    pairing: str = "sequential"       # sequential | unordered | grid
    grid_positions: list | None = None # (row, column) for each input image
    input_names: list | None = None
    disconnected: str = "largest"     # largest | reject
    refine: bool = False               # photometric (ECC) refinement
    reference: str = "middle"          # middle | first
    # Compositing
    exposure: bool = True
    blend_mode: str = "feather"        # feather | raw
    blend_levels: int = 6
    feather_power: float = 1.0
    crop: bool = True
    crop_margin: int = 0
    max_output_dim: int = 12000
    # Incremental "growing map" gate
    min_extend_ratio: float = 0.0    # min new bounding-box area as fraction of map
    min_extend_px: float = 0.0       # min new bounding-box area in px²
    # Assets / output
    weights_dir: str = "weights"
    viz: bool = True

    def __post_init__(self):
        from .catalog import resolve_options
        if self.engine == 'superpoint+superglue': self.engine = 'superglue'
        self.engine_options = resolve_options(self.engine, self.engine_options, asdict(self))
        for key, value in self.engine_options.items():
            if hasattr(self, key): setattr(self, key, value)
        if self.alignment not in ("homography", "translation"):
            raise ValueError("Alignment must be homography or translation")

    def to_dict(self):
        return asdict(self)

    def with_preset(self, preset: str) -> "StitchConfig":
        p = PRESETS.get((preset or "").lower())
        if p:
            for k, v in p.items():
                setattr(self, k, v)
        from .catalog import CATALOG, resolve_options
        self.engine_options.update(CATALOG[self.engine]['presets'].get(preset, {}))
        self.engine_options = resolve_options(self.engine,self.engine_options)
        for key,value in self.engine_options.items():
            if hasattr(self,key): setattr(self,key,value)
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
            try:
                b, g = load_image(p)
            except Exception as exc:
                self.reporter._send(dict(type='input_error',image=len(bgr)+1,
                    name=self.config.input_names[len(bgr)] if self.config.input_names else str(p),
                    reason=str(exc),code='image_load_failed'))
                raise
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
            "engine_options": cfg.engine_options,
            "weights": cfg.superglue_weights,
            "alignment": cfg.alignment,
            "input_names": cfg.input_names,
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
            from .contract import prepare_image
            fs = prepare_image(self.engine, images[i], cfg.input_names[i] if cfg.input_names else str(i+1))
            if cfg.viz and fs.small_gray is None:
                fs.small_gray, fs.scale = resize_to_max_dim(g, cfg.feature_max_dim)
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
        graph = None
        input_count = n
        if cfg.pairing in ("unordered", "grid"):
            from .unordered import align_unordered
            Hs, included, original_ref, graph = align_unordered(self.engine, features, cfg, rep)
            images = [images[i] for i in included]
            sizes = [sizes[i] for i in included]
            n = len(images)
            ref = included.index(original_ref)
        else:
            rep.stage_start("match", f"Matching {n - 1} pairs…")
            matches = []
            for i in range(n - 1):
                rep.stage_progress("match", (i + 1) / (n - 1),
                                   f"Pair {i + 1}↔{i + 2}")
                from .contract import match_pair
                matched = match_pair(self.engine,features[i],features[i+1])
                mkpts0,mkpts1,conf = matched.points0,matched.points1,matched.scores
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
            rep.stage_start("align", f"Estimating {cfg.alignment} transforms…")
            pairwise_H = []
            for i, (mkpts0, mkpts1, _conf) in enumerate(matches):
                rep.stage_progress("align", (i + 1) / (n - 1),
                                   f"{cfg.alignment.title()} {i + 1}→{i + 2}")
                if len(mkpts0) < 4:
                    raise StitchError(
                        f"Not enough matches ({len(mkpts0)}) between image "
                        f"{i + 1} and {i + 2}. Try a different engine or lower "
                        "the match threshold.")
                if cfg.alignment == "translation":
                    from .unordered import reliable_edge
                    H, _, reason = reliable_edge(mkpts0, mkpts1, sizes[i], sizes[i + 1],
                                                 cfg.ransac_thresh, alignment="translation")
                    if H is None:
                        raise StitchError(f"No reliable translation-only overlap between image {i + 1} and {i + 2}: {reason}. Images must have the same scale and orientation.")
                else:
                    H, _ = estimate_homography(mkpts0, mkpts1, cfg.ransac_thresh)
                if H is None:
                    raise StitchError(
                        f"Homography between image {i + 1} and {i + 2} is "
                        "degenerate (likely no overlap or repetitive scene).")
                if cfg.refine and cfg.alignment == "homography":
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
        blend_started = time.perf_counter()

        def blend_progress(fraction, message):
            elapsed = time.perf_counter() - blend_started
            rep.stage_progress("blend", fraction, f"{message} · {elapsed:.1f}s elapsed")

        rep.stage_start("blend", "Warping & multi-band blending…")
        blend_progress(0.0, "Computing canvas")
        offset, W, H, scale = blender.compute_canvas(
            Hs, sizes, max_output_dim=cfg.max_output_dim)
        rep.log(f"Blend canvas: {W}×{H} ({W * H / 1e6:.1f} MP), "
                f"{n} images, {cfg.blend_levels} pyramid levels; "
                f"warped images and masks: {n * W * H * 4 / 2**20:.0f} MiB",
                marker="blend")

        warped, masks = [], []
        for i, img in enumerate(images):
            blend_progress(0.05 + 0.3 * i / n, f"Warping image {i + 1}/{n}")
            M = blender.final_transform(Hs[i], offset, scale)
            w_img, w_mask = blender.warp_image(img, M, W, H)
            warped.append(w_img)
            masks.append(w_mask)
            blend_progress(0.05 + 0.3 * (i + 1) / n, f"Warped image {i + 1}/{n}")

        gains = np.ones(n, np.float32)
        if cfg.exposure and n > 1:
            blend_progress(0.35, "Exposure compensation")
            gains = blender.exposure_compensation(
                warped, masks,
                progress=lambda f, msg: blend_progress(0.35 + 0.2 * f, msg))

        blend_progress(0.55, f"Multi-band blending ({cfg.blend_levels} levels)")
        blended = blender.multi_band_blend(
            warped, masks, gains, cfg.blend_levels,
            progress=lambda f, msg: blend_progress(0.55 + 0.4 * f, msg))
        del warped
        blend_progress(0.95, "Preparing preview and coverage mask")

        if cfg.viz:
            preview, _ = resize_to_max_dim(blended, 1600)
            rep.image("blend", encode_b64(preview),
                      "Blended (uncropped)", f"{W}×{H} px",
                      {"width": W, "height": H})

        weight_sum = np.zeros((H, W), np.float32)
        for m in masks:
            weight_sum += m.astype(np.float32) / 255.0

        del masks
        rep.stage_done("blend",
                       f"Canvas {W}×{H} px in {time.perf_counter() - blend_started:.1f}s",
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
            "alignment": cfg.alignment,
            "num_images": n,
            "input_count": input_count,
            "graph": graph,
            "canvas": {"w": W, "h": H},
            "engine_options": cfg.engine_options,
            "device": str(getattr(self.engine,"device","cpu")),
            "output": {"w": result.shape[1], "h": result.shape[0]},
            "elapsed": round(time.time() - t0, 3),
            "reference": graph["reference"] if graph else ref + 1,
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
            Hs[i] = Hs[i + 1] @ pairwise_H[i]
        for i in range(ref + 1, n):
            Hs[i] = Hs[i - 1] @ np.linalg.inv(pairwise_H[i - 1])
        return Hs
