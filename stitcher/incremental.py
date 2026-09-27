"""Incremental "growing map" stitching.

Instead of feeding every image at once, the user starts with a single seed
image and then adds images one at a time.  Each addition is matched against the
*accumulated map* (not just the previous image), so the composite reliably grows
larger; if an image cannot be matched it is rejected without touching the map.

Memory stays bounded because only the map and the single new image are ever in
flight, blending is restricted to the new image's overlap region, and the map's
cached feature set is capped.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import blender
from .geometry import (estimate_homography, transform_points, warp_corners,
                       translation_matrix)
from .matching import FeatureSet, build_engine
from .progress import ProgressReporter
from .stitcher import StitchConfig, StitchError, NoExtensionError
from .utils import (encode_b64, resize_to_max_dim, draw_keypoints,
                    draw_matches)

# Safety cap: refuse to grow a map beyond this many pixels on its long side so
# a runaway sequence cannot exhaust memory.
MAX_MAP_DIM = 16000

# Cap on the number of keypoints kept in the accumulated map's feature set.
# SuperGlue attention is O(K^2); capping keeps matching fast as the map grows.
MAX_MAP_KEYPOINTS = 4000


class IncrementalStitcher:
    def __init__(self, config: StitchConfig | None = None,
                 reporter: ProgressReporter | None = None,
                 download_url: str | None = None):
        self.config = config or StitchConfig()
        self.reporter = reporter or ProgressReporter()
        self.download_url = download_url
        self.engine = None
        self._t0 = time.time()
        self.reset()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def reset(self):
        self.map_img = None            # BGR uint8 canvas
        self.map_mask = None           # uint8 0/255 valid mask
        self.map_weight = None         # float32 accumulated feather weight
        self.offset = np.array([0.0, 0.0])   # world origin in canvas coords
        self.map_features = None       # FeatureSet in *world* coordinates
        self.map_shape = None          # (H, W) world bbox for normalization
        self.count = 0
        self.placed = []
        self.pairwise_mode = self.config.engine not in ("sift","orb","superglue")
        self.engine = None
        # top-left of the most recently placed frame in map (canvas) coords,
        # plus its size -- what a live overlay needs to draw the viewport marker
        self.last_view = (0.0, 0.0)
        self.last_frame_size = (0, 0)

    @property
    def empty(self):
        return self.map_img is None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def seed(self, image, position=None):
        """Initialise the map from the first image (world == its pixels)."""
        from .preprocessing import preprocess_image
        image = preprocess_image(image, self.config.input_max_width, self.config.preprocessing)
        self.reset()
        rep = self.reporter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.engine = build_engine(self.config.to_dict())

        rep.plan({
            "engine": self.engine.name,
            "engine_options": self.config.engine_options,
            "device": str(getattr(self.engine,"device","cpu")),
            "weights": self.config.superglue_weights,
            "mode": "incremental",
            "alignment": self.config.alignment,
        })

        rep.stage_start("prepare", "Creating map from image 1")
        rep.stage_done("prepare", "Seed image loaded", {"image": 1})

        rep.stage_start("features", "Detecting features…")
        from .contract import prepare_image
        fs = prepare_image(self.engine,image,'Image 1') if self.pairwise_mode else self.engine.extract(gray)
        if self.config.viz:
            rep.image("features",
                      encode_b64(draw_keypoints(fs.small_gray,
                                                fs.keypoints * fs.scale)),
                      "Keypoints #1", f"{len(fs.keypoints)} keypoints",
                      {"image": 1, "keypoints": int(len(fs.keypoints))})
        rep.stage_done("features", f"{len(fs.keypoints)} keypoints")

        self.map_img = image.copy()
        self.map_mask = np.full(image.shape[:2], 255, np.uint8)
        self.map_weight = blender.feather_mask(self.map_mask,
                                               self.config.feather_power)
        self.offset = np.array([0.0, 0.0])
        self.map_features = fs
        self.placed = [dict(features=fs,transform=np.eye(3),position=position)]
        self.map_shape = gray.shape[:2]
        self.count = 1
        self.last_view = (0.0, 0.0)
        self.last_frame_size = (image.shape[1], image.shape[0])

        rep.stage_done("align", "Map initialized", {"map": self._map_dims()})
        rep.stage_done("blend", "Ready — add the next image")
        self._emit_result()
        return self.map_img

    def add(self, image, position=None):
        """Try to merge ``image`` into the map.  Raises :class:`StitchError`."""
        if self.empty:
            raise StitchError("Seed the map with a first image first.")

        from .preprocessing import preprocess_image
        image = preprocess_image(image, self.config.input_max_width, self.config.preprocessing)

        rep = self.reporter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        n = self.count + 1
        h_new, w_new = gray.shape[:2]

        # -- features -------------------------------------------------- #
        rep.stage_start("prepare", f"Loading image {n}")
        rep.stage_done("prepare", f"Image {n} loaded", {"image": n})

        rep.stage_start("features", f"Detecting features of image {n}…")
        from .contract import prepare_image
        fs_new = prepare_image(self.engine,image,f'Image {n}') if self.pairwise_mode else self.engine.extract(gray)
        if self.config.viz:
            rep.image("features",
                      encode_b64(draw_keypoints(fs_new.small_gray,
                                                fs_new.keypoints * fs_new.scale)),
                      f"Keypoints #{n}", f"{len(fs_new.keypoints)} keypoints",
                      {"image": n, "keypoints": int(len(fs_new.keypoints))})
        rep.stage_done("features", f"{len(fs_new.keypoints)} keypoints")

        # -- match against the map ------------------------------------- #
        rep.stage_start("match", f"Matching image {n} ↔ map…")
        proposed_H = None
        if self.pairwise_mode:
            from .contract import match_pair
            from .unordered import reliable_edge
            candidates = []
            for i, placed in enumerate(self.placed):
                previous = placed['position']
                if position is not None and previous is not None and sum(abs(a-b) for a,b in zip(position,previous)) != 1:
                    continue
                reference = placed['features']
                matched = match_pair(self.engine,fs_new,reference)
                H, count, reason = reliable_edge(matched.points0,matched.points1,fs_new.shape,reference.shape,
                                                self.config.ransac_thresh,self.config.alignment)
                rep.log(f'Image {n} ↔ {i+1}: {count}/{len(matched.points0)} inliers; {reason or "accepted"}',marker='match')
                if self.config.viz:
                    preview = draw_matches(fs_new.small_gray,reference.small_gray,
                        matched.points0*fs_new.scale,matched.points1*reference.scale,matched.scores,max_draw=500)
                    rep.image('match',encode_b64(preview),f'Matches {n} ↔ {i+1}',reason or 'Accepted overlap',
                        dict(pair_diagnostic=dict(pair=[n,i+1],matches=len(matched.points0),inliers=count,
                             accepted=H is not None,reason=reason,engine=self.config.engine,engine_options=self.config.engine_options),
                             names=[fs_new.image_id,reference.image_id],drawn_matches=min(len(matched.points0),500)))
                if H is not None:
                    candidates.append((count,placed['transform']@H,matched,placed['transform']))
            if not candidates:
                raise StitchError('No reliable overlap with a placed grid neighbor. Try a different model or input resolution.')
            candidates.sort(key=lambda item:item[0],reverse=True)
            _, proposed_H, matched, reference_H = candidates[0]
            if self.config.alignment == 'translation' and any(np.linalg.norm(candidate[1][:2,2]-proposed_H[:2,2])>2*self.config.ransac_thresh for candidate in candidates[1:]):
                raise StitchError('Neighbor matches disagree on the global translation; the map was not changed.')
            mkpts_new = matched.points0
            mkpts_map = transform_points(matched.points1,reference_H)
            conf = matched.scores
        else:
            mkpts_map, mkpts_new, conf = self.engine.match(self.map_features, fs_new)
        if len(mkpts_new) < 8:
            rep.stage_error("match", "Too few matches")
            raise StitchError(
                f"Could not match image {n} to the map "
                f"({len(mkpts_new)} matches). It may not overlap the current "
                "map, or try a different preset/engine.")
        if self.config.viz:
            self._emit_match_viz(fs_new, mkpts_new, mkpts_map, conf, n)
        rep.stage_done("match", f"{len(mkpts_new)} matches",
                       {"matches": int(len(mkpts_new))})

        # -- align + expand -------------------------------------------- #
        rep.stage_start("align", "Estimating transform & expanding map…")
        if proposed_H is not None:
            H = proposed_H
            mask = np.linalg.norm(transform_points(mkpts_new,H)-mkpts_map,axis=1) <= self.config.ransac_thresh
        elif self.config.alignment == "translation":
            from .unordered import reliable_edge
            H, _, reason = reliable_edge(
                mkpts_new, mkpts_map - self.offset, gray.shape[:2],
                self.map_img.shape[:2], self.config.ransac_thresh, "translation")
            if H is None:
                rep.stage_error("align", "No reliable translation-only overlap")
                raise StitchError(f"Could not place image {n} by translation: {reason}.")
            H = translation_matrix(*self.offset) @ H
            mask = np.linalg.norm(mkpts_new + H[:2, 2] - mkpts_map, axis=1) <= self.config.ransac_thresh
        else:
            H, mask = estimate_homography(mkpts_new, mkpts_map,
                                          self.config.ransac_thresh)
        if H is None or mask.sum() < 4:
            rep.stage_error("align", "Degenerate transform")
            raise StitchError(
                f"Could not estimate a stable transform for image {n} "
                f"(only {int(mask.sum()) if mask is not None else 0} inliers).")

        corners_world = warp_corners(H, w_new, h_new)
        Hm, Wm = self.map_img.shape[:2]
        cur_corners = np.array([
            [self.offset[0], self.offset[1]],
            [self.offset[0] + Wm, self.offset[1]],
            [self.offset[0] + Wm, self.offset[1] + Hm],
            [self.offset[0], self.offset[1] + Hm],
        ], np.float64)
        all_pts = np.concatenate([corners_world, cur_corners], axis=0)
        xmin = int(np.floor(all_pts[:, 0].min()))
        xmax = int(np.ceil(all_pts[:, 0].max()))
        ymin = int(np.floor(all_pts[:, 1].min()))
        ymax = int(np.ceil(all_pts[:, 1].max()))
        new_offset = np.array([float(xmin), float(ymin)])
        new_W, new_H = xmax - xmin, ymax - ymin

        if max(new_W, new_H) > MAX_MAP_DIM:
            rep.stage_error("align", "Map too large")
            raise StitchError(
                f"Map would become {new_W}×{new_H} px — too large. "
                "Start a new map or use a smaller preset.")

        # ---- minimum-extension gate --------------------------------- #
        # Skip frames that don't meaningfully grow the map (e.g. a stationary
        # camera re-sending near-identical views).
        old_area = float(Wm) * float(Hm)
        new_area = float(new_W) * float(new_H)
        added_area = new_area - old_area
        required = max(self.config.min_extend_px,
                       self.config.min_extend_ratio * old_area)
        if added_area < required:
            rep.stage_error("align", "No meaningful extension")
            raise NoExtensionError(
                f"Image {n} adds only {int(added_area)} px² of new area "
                f"(need ≥ {int(required)} px²) — skipped.",
                added_area=added_area, required_area=required)

        # Re-home the existing map into the (possibly larger) canvas.
        dx = int(round(self.offset[0] - new_offset[0]))
        dy = int(round(self.offset[1] - new_offset[1]))
        new_map = np.zeros((new_H, new_W, 3), np.uint8)
        new_mask = np.zeros((new_H, new_W), np.uint8)
        new_weight = np.zeros((new_H, new_W), np.float32)
        new_map[dy:dy + Hm, dx:dx + Wm] = self.map_img
        new_mask[dy:dy + Hm, dx:dx + Wm] = self.map_mask
        new_weight[dy:dy + Hm, dx:dx + Wm] = self.map_weight

        # Warp the new image into the (world -> canvas) frame.
        M = translation_matrix(-new_offset[0], -new_offset[1]) @ H
        warped, wmask = blender.warp_image(image, M, new_W, new_H)

        rep.log(f"inliers {int(mask.sum())}/{len(mkpts_new)}; "
                f"canvas {new_W}×{new_H}", "info", "align")
        rep.stage_done("align", f"Canvas {new_W}×{new_H} px",
                       {"canvas": {"w": new_W, "h": new_H},
                        "inliers": int(mask.sum())})

        # -- blend ----------------------------------------------------- #
        rep.stage_start("blend", "Blending into map…")
        if self.config.exposure:
            g = self._estimate_gain(new_map, new_mask, warped, wmask)
            warped = np.clip(warped.astype(np.float32) * g, 0, 255).astype(np.uint8)
        if self.config.blend_mode == "raw":
            new_map, new_mask = self._blend_into_map(new_map, new_mask, warped,
                                                     wmask, self.config.blend_levels)
            new_weight = (new_mask > 0).astype(np.float32)
        else:
            new_feather = blender.feather_mask(wmask, self.config.feather_power)
            new_map, new_weight = self._blend_feather_into_map(
                new_map, new_weight, warped, new_feather)
            new_mask = (new_weight > 0).astype(np.uint8) * 255
        rep.stage_done("blend", "Blended")

        # -- commit state ---------------------------------------------- #
        self.map_img = new_map
        self.map_mask = new_mask
        self.map_weight = new_weight
        self.offset = new_offset
        self.map_shape = (new_H, new_W)

        if self.pairwise_mode:
            self.placed.append(dict(features=fs_new,transform=H.copy(),position=position))
        else:
            kpts_world = transform_points(fs_new.keypoints, H)
            self.map_features = self._merge_features(
                self.map_features,
                FeatureSet(keypoints=kpts_world.astype(np.float32),
                           descriptors=fs_new.descriptors,scores=fs_new.scores,shape=(new_H,new_W)))
        self.count = n
        # where the freshly placed frame landed in the (new) map's canvas coords
        self.last_view = (
            float(corners_world[:, 0].min() - new_offset[0]),
            float(corners_world[:, 1].min() - new_offset[1]),
        )
        self.last_frame_size = (w_new, h_new)
        self._emit_result()
        return self.map_img

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _map_dims(self):
        h, w = self.map_img.shape[:2]
        return {"w": int(w), "h": int(h)}

    def _emit_result(self):
        preview, _ = resize_to_max_dim(self.map_img, 2000)
        meta = {
            "input_max_width": self.config.input_max_width,
            "preprocessing": self.config.preprocessing,
            "engine": self.engine.name,
            "engine_options": self.config.engine_options,
            "device": str(getattr(self.engine,"device","cpu")),
            "num_images": self.count,
            "output": self._map_dims(),
            "elapsed": round(time.time() - self._t0, 3),
            "mode": "incremental",
            "alignment": self.config.alignment,
            "download": self.download_url,
        }
        self.reporter.result(encode_b64(preview, ".jpg", 92),
                             self.map_img.shape[1], self.map_img.shape[0], meta)

    def _emit_match_viz(self, fs_new, mkpts_new, mkpts_map, conf, n):
        map_gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
        map_prev, mscale = resize_to_max_dim(map_gray, 900)
        mk_map_canvas = mkpts_map - self.offset
        viz = draw_matches(fs_new.small_gray, map_prev,
                           mkpts_new * fs_new.scale, mk_map_canvas * mscale,
                           conf)
        self.reporter.image("match", encode_b64(viz),
                            f"Matches #{n} → map", f"{len(mkpts_new)} matches",
                            {"image": n, "matches": int(len(mkpts_new))})

    @staticmethod
    def _estimate_gain(map_img, map_mask, new_img, new_mask):
        inter = (map_mask > 0) & (new_mask > 0)
        if inter.sum() < 200:
            return 1.0
        m = np.median(map_img[inter])
        nn = np.median(new_img[inter])
        if nn < 1.0:
            return 1.0
        return float(np.clip(m / nn, 1.0 / 1.6, 1.6))

    @staticmethod
    def _blend_into_map(map_img, map_mask, new_img, new_mask, levels=6):
        """Raw paste: multi-band blend only the overlap box, hard-paste the rest."""
        H, W = map_img.shape[:2]
        overlap = (map_mask > 0) & (new_mask > 0)
        out = map_img.copy()
        out_mask = ((map_mask > 0) | (new_mask > 0)).astype(np.uint8) * 255

        if not overlap.any():
            out[new_mask > 0] = new_img[new_mask > 0]
            return out, out_mask

        ys, xs = np.where(overlap)
        pad = 6
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(H, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(W, int(xs.max()) + 1 + pad)

        m_crop = map_img[y0:y1, x0:x1]
        n_crop = new_img[y0:y1, x0:x1]
        mm = (map_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        nm = (new_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        levels = max(1, min(levels, int(np.log2(min(y1 - y0, x1 - x0))) - 2))
        blended = blender.multi_band_blend([m_crop, n_crop], [mm, nm],
                                           np.ones(2, np.float32), levels)
        out[y0:y1, x0:x1] = blended

        outside = np.ones((H, W), bool)
        outside[y0:y1, x0:x1] = False
        paste = (new_mask > 0) & outside
        out[paste] = new_img[paste]
        return out, out_mask

    @staticmethod
    def _blend_feather_into_map(map_img, map_weight, new_img, new_feather):
        """Feather-accumulate ``new_img`` into ``map_img`` (scan-mode style).

        Every pixel is the normalized weighted average of every frame covering
        it, using a distance-transform feather per frame, so there are no hard
        seams at the overlap boundary.  ``map_weight`` is the accumulated
        float32 weight and ``new_feather`` is the new frame's feather weight.
        """
        acc = map_img.astype(np.float32) * map_weight[..., None]
        acc += new_img.astype(np.float32) * new_feather[..., None]
        weight = map_weight + new_feather
        out = np.clip(acc / np.maximum(weight[..., None], 1e-6),
                      0, 255).astype(np.uint8)
        return out, weight

    @staticmethod
    def _merge_features(map_fs, new_fs, cap=MAX_MAP_KEYPOINTS):
        k = np.concatenate([map_fs.keypoints, new_fs.keypoints])
        d = np.concatenate([map_fs.descriptors, new_fs.descriptors])
        s = np.concatenate([map_fs.scores, new_fs.scores])
        if len(k) > cap:
            idx = np.argsort(s)[::-1][:cap]
            k, d, s = k[idx], d[idx], s[idx]
        return FeatureSet(keypoints=k, descriptors=d, scores=s,
                          shape=map_fs.shape)
