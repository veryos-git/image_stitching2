#!/usr/bin/env python3
"""Download the SuperPoint + SuperGlue pretrained weights.

The checkpoints are served from the official repository (GitHub raw, which
streams the LFS objects).  Run once before first use:

    python download_weights.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

BASE = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights"

FILES = {
    "superpoint_v1.pth": 5_206_086,
    "superglue_indoor.pth": 48_233_807,
    "superglue_outdoor.pth": 48_233_807,
}


def download(url, dest, expected_size):
    print(f"  -> {dest.name}", flush=True)
    with urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    if expected_size and total != expected_size:
        raise RuntimeError(
            f"size mismatch for {dest.name}: got {total}, expected {expected_size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights")
    ap.add_argument("--indoor-only", action="store_true")
    ap.add_argument("--outdoor-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = dict(FILES)
    if args.indoor_only:
        files.pop("superglue_outdoor.pth")
    if args.outdoor_only:
        files.pop("superglue_indoor.pth")

    ok = True
    for name, size in files.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"[skip] {name} already present", flush=True)
            continue
        print(f"[get]  {name}", flush=True)
        try:
            download(f"{BASE}/{name}", dest, size)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}", flush=True)
            ok = False
    print("done" if ok else "one or more downloads failed", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
