#!/usr/bin/env python3
"""Command-line panorama stitcher.

Example::

    python run_stitcher.py --input img1.jpg img2.jpg img3.jpg -o pano.jpg
    python run_stitcher.py --input-dir ./photos --engine orb -o pano.jpg
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from stitcher import PanoramaStitcher, StitchConfig, ProgressReporter
from stitcher.utils import discover_images


def _console_emit(event):
    t = event.get("type")
    if t == "step":
        status = event["status"]
        marker = event["marker"]
        msg = event.get("message", "")
        bar = "#" * int(event.get("progress", 0) * 40)
        icon = {"running": "…", "done": "✓", "error": "✗"}.get(status, "·")
        if status == "running":
            print(f"\r[{event['progress']*100:5.1f}%] {icon} {marker:<9} {msg[:70]:<70}",
                  end="", flush=True)
        else:
            print(f"\r[{event['progress']*100:5.1f}%] {icon} {marker:<9} {msg[:70]:<70}")
    elif t == "log":
        print(f"      · {event['message']}")
    elif t == "image":
        print(f"      🖼  {event['label']}: {event.get('caption','')}")
    elif t == "error":
        print(f"\nERROR: {event['message']}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="AI panorama stitcher (SuperGlue)")
    ap.add_argument("--input", nargs="+", help="image files to stitch")
    ap.add_argument("--input-dir", help="stitch all images in a directory")
    ap.add_argument("-o", "--output", default="panorama.jpg")
    ap.add_argument("--engine", default="superglue",
                    choices=["superglue", "sift", "orb"])
    ap.add_argument("--weights", default="outdoor", choices=["indoor", "outdoor"])
    ap.add_argument("--preset", default="balanced",
                    choices=["fast", "balanced", "best"],
                    help="speed/quality preset (sets feature-max-dim, "
                         "max-keypoints, sinkhorn, match-threshold)")
    ap.add_argument("--max-keypoints", type=int, default=None)
    ap.add_argument("--match-threshold", type=float, default=None)
    ap.add_argument("--feature-max-dim", type=int, default=None)
    ap.add_argument("--sinkhorn", type=int, default=None)
    ap.add_argument("--ransac-thresh", type=float, default=3.0)
    ap.add_argument("--reference", default="middle", choices=["middle", "first"])
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--blend-levels", type=int, default=6)
    ap.add_argument("--no-exposure", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--weights-dir", default="weights")
    args = ap.parse_args()

    paths = args.input or []
    if args.input_dir:
        paths += discover_images(args.input_dir)
    if not paths:
        ap.error("provide --input FILES or --input-dir DIR")

    cfg = StitchConfig(
        engine=args.engine,
        superglue_weights=args.weights,
        ransac_thresh=args.ransac_thresh,
        reference=args.reference,
        refine=not args.no_refine,
        blend_levels=args.blend_levels,
        exposure=not args.no_exposure,
        crop=not args.no_crop,
        viz=not args.no_viz,
        weights_dir=args.weights_dir,
    ).with_preset(args.preset)

    # Explicit flags override the preset.
    if args.max_keypoints is not None:
        cfg.max_keypoints = args.max_keypoints
    if args.match_threshold is not None:
        cfg.match_threshold = args.match_threshold
    if args.feature_max_dim is not None:
        cfg.feature_max_dim = args.feature_max_dim
    if args.sinkhorn is not None:
        cfg.sinkhorn_iterations = args.sinkhorn

    reporter = ProgressReporter(emit=_console_emit)
    stitcher = PanoramaStitcher(cfg, reporter)

    t0 = time.time()
    try:
        result = stitcher.stitch_paths(paths)
    except Exception as exc:  # noqa: BLE001
        reporter.error(str(exc))
        print(f"\nFailed: {exc}", file=sys.stderr)
        sys.exit(1)

    ok = cv2.imwrite(args.output, result.panorama,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        print(f"\nCould not write {args.output}", file=sys.stderr)
        sys.exit(1)
    print(f"\nWrote {args.output} "
          f"({result.panorama.shape[1]}x{result.panorama.shape[0]}) "
          f"in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
