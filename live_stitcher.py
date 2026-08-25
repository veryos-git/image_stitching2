#!/usr/bin/env python3
"""Standalone, headless example of the live "growing map" stitcher.

This is the piece you copy into your own project: it shows how to embed
:class:`IncrementalStitcher` in a continuous-feed loop and gate additions so a
(nearly) stationary stream only updates the map when a frame actually extends
it.

    from live_stitcher import LivePanorama

    pano = LivePanorama(min_extend_ratio=0.01, min_extend_px=20000)

    for bgr_frame in your_camera_or_stream():
        status, updated_map, msg = pano.feed(bgr_frame)
        if status in ("seeded", "added"):
            save(updated_map)      # only when the map changed

``feed`` returns one of:
  * ``"seeded"``   — first frame, map created;
  * ``"added"``    — frame extended the map;
  * ``"skipped"``  — frame matched but added too little new area (redundant);
  * ``"rejected"`` — frame couldn't be matched / transform failed.

Only ``"seeded"`` and ``"added"`` return an updated map image.

Run a folder-watch demo (stand-in for a live camera):

    python live_stitcher.py --dir ./frames --out growing_map.jpg \
        --min-extend-ratio 0.01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from stitcher import (IncrementalStitcher, StitchConfig, StitchError,
                      NoExtensionError, ProgressReporter)
from stitcher.utils import discover_images


class LivePanorama:
    """Thin wrapper around :class:`IncrementalStitcher` for live streams."""

    def __init__(self, reporter=None, **overrides):
        cfg = StitchConfig(**{
            "engine": "superglue",
            "superglue_weights": "outdoor",
            "feature_max_dim": 512,      # fast: features on a downscaled copy
            "max_keypoints": 512,
            "sinkhorn_iterations": 20,
            "min_extend_ratio": 0.01,    # skip frames adding <1% new bbox area
            "min_extend_px": 20000,      # ...and <20k px² absolute
            "viz": False,
            **overrides,
        })
        self.stitcher = IncrementalStitcher(
            cfg, reporter=reporter or ProgressReporter(emit=lambda e: None))

    def feed(self, bgr):
        """Feed one BGR frame; returns ``(status, map_or_None, message)``."""
        if self.stitcher.empty:
            self.stitcher.seed(bgr)
            return "seeded", self.stitcher.map_img, "map started"
        try:
            self.stitcher.add(bgr)
            return "added", self.stitcher.map_img, \
                f"map extended to {self.stitcher.map_img.shape[1]}x{self.stitcher.map_img.shape[0]}"
        except NoExtensionError as exc:
            return "skipped", None, str(exc)
        except StitchError as exc:
            return "rejected", None, str(exc)

    def current(self):
        return self.stitcher.map_img


def _console_reporter(event):
    if event["type"] == "step" and event["status"] == "done":
        print(f"  ✓ {event['marker']}: {event['message']}")
    elif event["type"] == "result":
        print(f"  🗺 map {event['width']}x{event['height']} "
              f"({event['meta']['num_images']} images)")


def main():
    ap = argparse.ArgumentParser(description="Live incremental panorama demo")
    ap.add_argument("--dir", required=True, help="folder to watch for images")
    ap.add_argument("--out", default="growing_map.jpg")
    ap.add_argument("--min-extend-ratio", type=float, default=0.01)
    ap.add_argument("--min-extend-px", type=float, default=20000)
    ap.add_argument("--engine", default="superglue",
                    choices=["superglue", "sift", "orb"])
    ap.add_argument("--poll", type=float, default=1.0,
                    help="seconds between folder polls")
    args = ap.parse_args()

    folder = Path(args.dir)
    pano = LivePanorama(
        reporter=ProgressReporter(emit=_console_reporter),
        engine=args.engine,
        min_extend_ratio=args.min_extend_ratio,
        min_extend_px=args.min_extend_px,
    )

    print(f"Watching {folder} for new images (Ctrl-C to stop)…")
    seen = set()
    try:
        while True:
            for path in discover_images(folder):
                if path in seen:
                    continue
                seen.add(path)
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                status, updated, msg = pano.feed(bgr)
                print(f"{Path(path).name}: {status} — {msg}")
                if updated is not None:
                    cv2.imwrite(args.out, updated,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nstopped.")
    if pano.current() is not None:
        cv2.imwrite(args.out, pano.current(),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"final map -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
