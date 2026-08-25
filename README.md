# ⛰️ AI Panorama Stitcher

A powerful and fast **image stitching / mosaicing / panorama** engine powered by
[SuperPoint](https://arxiv.org/abs/1712.07629) + [SuperGlue](https://arxiv.org/abs/1911.11763)
(CVPR 2020, Magic Leap) feature matching, wrapped in a **web app that streams
every pipeline step in real time with markers**.

```
upload → SuperPoint → SuperGlue → RANSAC homography → exposure → multi-band blend → crop
          (features)   (matching)      (alignment)                            (composite)
```

## Highlights

- **Two modes** — *Batch* (select all images, stitch in one go) and
  *Incremental* (seed with one image, then add images one-by-one: the map
  grows, and images that can't be matched are rejected without damage).
- **AI feature matching** — SuperPoint keypoint detection + SuperGlue graph
  neural matching for robust correspondences even under large rotation,
  scale and viewpoint change.
- **Fast fallbacks** — one flag switches to classic **SIFT** or **ORB** (≈1s
  CPU) when you don't need the heavy network.
- **Production-grade compositing** — RANSAC-robust homographies, optional
  photometric (ECC) refinement, exposure compensation and **multi-band
  (Laplacian pyramid) blending** for invisible seams, plus automatic black
  border cropping (largest inscribed rectangle).
- **Realtime markers** — the whole pipeline emits structured progress events
  (`plan / step / image / log / result`) consumed identically by the CLI and
  the web UI.

---

## Installation

```bash
cd image_stitching
python3 -m venv .venv
source .venv/bin/activate

pip install numpy opencv-python-headless fastapi "uvicorn[standard]" python-multipart
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU (recommended here)
# or, for a GPU build:  pip install torch

python download_weights.py     # SuperPoint + SuperGlue checkpoints (~100 MB)
```

> The weights are also bundled under `weights/` if you received this project
> already prepared.

## Command line

```bash
# basic
python run_stitcher.py --input img1.jpg img2.jpg img3.jpg -o pano.jpg

# a whole folder, in filename order
python run_stitcher.py --input-dir ./photos -o pano.jpg

# full control
python run_stitcher.py \
  --input a.jpg b.jpg c.jpg \
  --engine superglue --weights outdoor \
  --preset best \
  --ransac-thresh 3.0 --reference middle \
  --blend-levels 6 --no-exposure --no-crop \
  -o pano.jpg

# quick preview: heavy downscale for speed
python run_stitcher.py --input a.jpg b.jpg c.jpg --preset fast -o pano.jpg
```

## Speed

The AI matcher's cost is dominated by two things, both controlled by presets:

| Preset | feature_max_dim | max_keypoints | sinkhorn | relative speed |
| --- | --- | --- | --- | --- |
| `fast` | 512 px | 512 | 20 | ~4× |
| `balanced` (default) | 1024 px | 1024 | 50 | 1× |
| `best` | 1600 px | 2048 | 100 | ~0.4× |

`feature_max_dim` is the biggest lever: SuperPoint runs on the image
**downscaled** to that many pixels, but the final panorama is still rendered at
full resolution — so lowering it speeds up matching without shrinking the
output. On CPU, going 1600→512 px is ~8× faster for feature extraction alone.
SuperGlue's attention scales as O(K²), so fewer keypoints also helps. Torch is
pinned to 4 threads by default (small conv/attention ops run faster there than
across all cores).

## Web app

```bash
python server.py              # http://127.0.0.1:8000
python server.py --port 8080
```

**Batch mode** — drop 2+ images (drag to reorder), pick options, hit
**Stitch panorama**. The right pane live-updates a marker timeline, the global
progress bar, keypoint / match / blended previews, and the final downloadable
panorama.

**Incremental mode** — pick *Incremental*, choose one seed image, then add
images one by one. Each addition is matched against the accumulated map and
blended in, so the map visibly grows; if an image can't be matched it is
rejected with a message and the map is left untouched.

The backend exposes:

| Route | Purpose |
| --- | --- |
| `POST /api/stitch` | batch upload + options → `{job_id}` |
| `WS /ws/{job_id}` | live progress stream (replays history on reconnect) |
| `GET /api/jobs/{id}/result` | batch final JPEG |
| `POST /api/incremental/start` | seed an incremental map (first image) |
| `POST /api/incremental/add` | add one image to a session → growing map |
| `GET /api/incremental/{sid}/map.jpg` | current full-res map |
| `DELETE /api/incremental/{sid}` | discard a session |
| `POST /api/projects` | save a completed session (`{job_id, name}`) |
| `GET /api/projects` | list saved projects |
| `GET /api/projects/{id}` | project metadata + full event history (replayable) |
| `POST /api/projects/{id}/rename` | rename a project |
| `DELETE /api/projects/{id}` | delete a project |
| `GET /api/projects/{id}/files/{path}` | serve result / intermediate images |
| `GET /api/health` | liveness |

## Projects (persisted sessions)

Every session can be saved as a **project** from the web UI (or
`POST /api/projects`). A project is a directory under `projects/<id>/` holding
the original inputs, the final `result.jpg`, every intermediate visualization,
and the full marker event history — so you can reopen it later and replay the
whole timeline exactly as it happened, or re-run it with different options.

## Standalone / live integration

The core is a plain Python library with **no web dependency** — just `numpy`,
`opencv-python-headless` and `torch` (plus the downloaded weights). Import it
directly from any Python process:

```python
from stitcher import IncrementalStitcher, StitchConfig, StitchError, NoExtensionError

cfg = StitchConfig(engine="superglue", superglue_weights="outdoor",
                   feature_max_dim=512, max_keypoints=512,
                   min_extend_ratio=0.01,   # gate: skip frames adding <1% area
                   min_extend_px=20000)     # ...or fewer than 20k px²

stitcher = IncrementalStitcher(cfg)          # headless: reporter defaults to no-op

def feed(bgr_frame):
    if stitcher.empty:
        stitcher.seed(bgr_frame)
        return "seeded", stitcher.map_img
    try:
        stitcher.add(bgr_frame)
        return "added", stitcher.map_img
    except NoExtensionError:
        return "skipped", None        # matched, but didn't extend the map
    except StitchError as e:
        return "rejected", None       # couldn't match / transform failed
```

A ready-to-run wrapper lives in **`live_stitcher.py`** (`LivePanorama` class +
a folder-watching demo), and `run_stitcher.py` is the batch CLI. Pass an
`emit` callback to `ProgressReporter` to receive the same marker events the web
app shows.

**Minimum-extension gate** — for a continuous/stationary feed, set
`min_extend_ratio` / `min_extend_px` on the config. `add()` then raises
`NoExtensionError` (instead of re-blending) when a frame's warped footprint
extends the current map's bounding box by less than the threshold, so
near-identical frames are skipped cheaply.

## Key options

| Option | Default | Notes |
| --- | --- | --- |
| `--engine` | `superglue` | `superglue` · `sift` · `orb` |
| `--weights` | `outdoor` | SuperGlue `indoor` (ScanNet) / `outdoor` (MegaDepth) |
| `--preset` | `balanced` | `fast` · `balanced` · `best` speed/quality presets |
| `--max-keypoints` | `1024` | raise for harder scenes (up to `2048+`) |
| `--match-threshold` | `0.2` | lower → more matches, more outliers |
| `--feature-max-dim` | `1024` | resize before inference (speed vs. detail) |
| `--sinkhorn` | `50` | SuperGlue optimal-transport iterations |
| `--ransac-thresh` | `3.0` | px reprojection threshold |
| `--reference` | `middle` | which image anchors the global frame |
| `--blend-levels` | `6` | multi-band pyramid depth |
| `min_extend_ratio` | `0.0` | incremental gate: min new bbox area (fraction of map) |
| `min_extend_px` | `0.0` | incremental gate: min new bbox area (px²) |
| `--no-exposure` | off | disable gain equalisation |
| `--no-crop` | off | keep full canvas |

## Progress event schema

Every message is JSON-able; `marker` is one of
`prepare, features, match, align, blend, crop`.

```jsonc
{"type":"plan",  "stages":[{"marker":"features","label":"Detect Features","weight":0.3}, …]}
{"type":"step",  "marker":"match","status":"running","message":"Pair 1↔2","progress":0.47,"fraction":0.5,"detail":{}}
{"type":"image", "marker":"match","label":"Matches 1↔2","image":"data:image/jpeg;base64,…","caption":"196 matches"}
{"type":"log",   "level":"info","message":"H 1→2: 138/196 inliers"}
{"type":"result","image":"data:…","width":1496,"height":478,"meta":{…}}
{"type":"end"}
```

## Project layout

```
stitcher/
  models.py      SuperPoint + SuperGlue (PyTorch 2.x, CPU/GPU)
  matching.py    SuperGlueEngine + ClassicEngine (SIFT/ORB) abstraction
  geometry.py    homography RANSAC, ECC refinement, composition
  blender.py     canvas, warping, exposure compensation, multi-band blend, crop
  stitcher.py    PanoramaStitcher orchestrator + StitchConfig (batch)
  incremental.py IncrementalStitcher (growing-map mode)
  progress.py    marker-based ProgressReporter (event schema)
server.py        FastAPI + WebSocket backend
static/          web UI (index.html, app.js, style.css)
run_stitcher.py  CLI
download_weights.py
```

## Notes

- **GPU vs CPU** — `torch.cuda.is_available()` is used automatically; on CPU the
  `balanced` preset keeps a 3-image panorama at ~4 s and `fast` at ~2 s.
- **Indoor vs outdoor** — use `outdoor` for city/landscape panoramas, `indoor`
  for close-range room scenes (matches the SuperGlue training domains).
- The core library is independent of the web layer: import
  `PanoramaStitcher` + `ProgressReporter` and pass your own `emit` callback to
  drive any frontend.

## Credits

SuperPoint & SuperGlue: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
Andrew Rabinovich — [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork).
# image_stitching2
