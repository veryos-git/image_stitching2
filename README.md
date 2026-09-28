# Start all tools

From this project folder, run:

```bash
./start
```

Open **http://127.0.0.1:8001/** to go directly to the classic workspace.
The console also prints the overview URL (`/overview`) and links to all four tools. The overview page links to pipeline comparison,
microscope mosaicing, the classic batch/incremental stitcher, and guided stitching.

The launcher creates `.venv`, installs all app requirements (including the
comparison and microscope packages), and sets up XFeat, SuperPoint/SuperGlue,
EfficientLoFTR and MatchAnything automatically. You need Python with `venv`
support and Git installed. First setup needs internet access and downloads large
packages and model checkpoints; allow several GB of disk space. Setup failures
stop startup with an error; rerun `./start` after fixing the reported problem.
Successful setup is cached in `.venv` and repeated when the launcher, requirements,
or setup scripts change, or required model files are missing. Other comparison
checkpoints download when first selected. Optional MASt3R and PixelStitch research
integrations still require their documented separate setup.

The launcher reuses an already-running Stitch Lab and selects the next free port
if another application occupies the requested port. Stop a newly started server
with **Ctrl+C**. To choose a port explicitly:
`./start --port 9000`.

A separate grid-aware microscope mosaicer is available at `/microscope`.
See [MICROSCOPE.md](MICROSCOPE.md) for its CLI, manifest, TIFF exports,
PixelStitch integration and validation limits.

# Learned models in the classic workspace

Classic also offers four **Optional batch corrections**, all off by default:

- **Estimate and correct illumination shading** estimates a smooth, per-channel flat field from up to 48 equal-sized original tiles, then corrects all tiles before input resizing and matching. It works in both batch modes with Original colors, and runs only once before row stitching. Recurring specimen structures can bias the estimate; it is not a substitute for a calibrated flat field.
- **Optimize all measured grid translations together** replaces spanning-tree placement with six Huber-reweighted least-squares iterations over accepted translation constraints. Edges are weighted by inlier count; loop residuals and resulting positions are saved in `diagnostics/graph.json`.
- **Retry failed overlaps with the selected matcher** predicts overlap from connected measured paths or at least two consistent measured steps in the same grid direction. It reruns the same feature matcher on overlap crops, restores full processed-tile coordinates, and applies the existing geometric checks plus a prediction-distance check. Retry artifacts are retained separately under `diagnostics/guided/`. No NCC matcher or automatic model substitution is introduced.
- **Allow predicted placement for missing grid links** adds explicitly marked low-weight (`0.02`) constraints between measured components when at least two agreeing grid steps support the prediction and the predicted tiles overlap. Outputs disclose predicted links; diagnostics retain them separately from accepted measured pairs. With no adequate measured support, the grid still fails rather than assigning arbitrary positions.

The three graph options require **Grid neighbors (single pass)** and **Translation only**. They can be enabled independently. They do not apply to rows-first or incremental stitching. Saved projects restore all four settings; older projects default them to off. Exposure compensation remains its existing independent checkbox. These options may increase runtime and memory use, and do not guarantee that 256 px inputs will stitch successfully. To retain original tile detail, set **Input width** to `0`; the separate model matching-resolution setting still controls feature inference.

Validation: `python -m unittest discover -s tests -p 'test_enhancements.py' -v` covers loop adjustment, weak priors, guided crop coordinates and recovery, rejection of unsupported predictions, flat-field behavior, API validation and saved settings. With an isolated server on port 8769 and Chrome/Playwright installed, `python tests/browser_enhancements.py` checks the controls, a batch run, and project restoration.

Opening `/` redirects to `/classic`; all tools remain listed at `/overview`.
Classic defaults to **DISK + LightGlue**, **translation only (XY)**,
**50 px RANSAC** (maximum **100 px**), and **Rows first, then combine**.

**Input width** defaults to **1080 pixels wide**. Images wider than this are
downscaled before matching and blending, preserving aspect ratio; smaller images
are never enlarged. Enter another width or **0** to retain original resolution.
**Input processing → Sobel edges** optionally replaces each resized image with
its Sobel gradient magnitude (horizontal and vertical edges). The output is an
edge-image panorama. Preprocessing runs once on each original tile, before either
stitching stage, and also applies to incremental inputs. Settings are saved with
projects; incremental settings stay fixed until the map is reset.

`/classic` and `/compare` share one model catalog. Search by model/family, filter
installed pipelines or methods without SuperPoint, and select model-specific
settings independently of translation/homography alignment. Research components
and unimplemented families stay visible with their blocking reasons.

Installed learned alternatives require **Verify model inference** before they can
start a job. Verification may download upstream weights and runs real inference
on unequal crops of a microscope tile with a known shift, plus an unrelated
negative. Results identify the runtime, device and loaded weight hash; installation
alone is not called readiness. CPU verification does not validate CUDA. CLI:

```bash
.venv/bin/python verify_engines.py xfeat aliked-lightglue efficientloftr --device cpu
```

The classic pair inspector offers all/inlier/outlier/confidence previews and
**Retry this pair with… / Compare selected models**. Retries save separate artifacts
and leave the accepted mosaic constraints unchanged. Rerun the grid to apply a new
model. Full bounded correspondences, geometric masks, residuals, options and
preprocessing are stored beside each job and copied into saved projects. Reloading
a project restores its inputs, positions and resolved model settings.

See [implementation and validation](validation/classic_ml/README.md) for the
implemented scope, blocked research families, test commands and measured results.
The five tested 512 px configurations did not connect the problematic 77-tile scan;
model integration alone does not guarantee registration success.

# Stitch Lab — pipeline comparison

The `/compare` web page compares matching pipelines using identical images and
shared RANSAC/compositing settings. The original batch/incremental workspace
remains at `/static/index.html`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-comparison.txt
# For CPU-only PyTorch, install torch/torchvision from the PyTorch CPU index first.
git clone https://github.com/verlab/accelerated_features vendor/accelerated_features
.venv/bin/python server.py --port 8001
```

Open http://127.0.0.1:8001/compare. Three microscope demo tiles load by default;
choose six tiles or the full 36-tile scan, or upload 2–36 images. Uploaded images
can be reordered. The full demo uses snake ordering to maintain adjacency.
Use **Shuffle current images** to test unknown ordering, or **Random images** to
sample tiles from anywhere in the scan. Both select **Unordered · find overlaps**
automatically. Random selections can contain missing links or entirely unrelated
tiles; more images do not guarantee a connected panorama. The input names/order
and random sample seed are saved in the report.

Sequential mode requires overlap between adjacent inputs. Unordered mode tests
all N(N−1)/2 pairs (630 for 36 images), validates homographies using at least 12
inliers, 25% inlier ratio, spatial coverage and footprint/overlap checks, then
places images along a maximum-inlier spanning tree. It works with uploads too.
Choose **Largest connected group** for an explicitly labeled partial panorama,
or **Require all images to connect** to reject disconnected sets. Reports show
overlap groups, excluded input numbers, pair diagnostics and tree edges. Zero
reliable overlaps produce a clear failure rather than an invented arrangement.
Dense matchers can be expensive in this mode. The graph uses homographies and
heuristic rejection; repetitive textures can still produce false connections.
It does not perform global bundle adjustment, loop-closure optimization or
non-planar 3D reconstruction.

Available adapters: SIFT, ORB, XFeat, XFeat semi-dense, XFeat + LighterGlue,
ALIKED/DISK/SIFT + LightGlue, indoor/outdoor LoFTR, RoMa, Tiny RoMa,
MASt3R, and the original SuperPoint + SuperGlue baseline. XFeat uses the
upstream **LighterGlue** checkpoint, not a generic LightGlue checkpoint.

Missing dependencies and unverified learned alternatives are disabled with setup
instructions and explicit inference verification. Dependencies being present do
not guarantee checkpoint availability. Downloads happen on verification or model
use, with errors shown per pipeline; other selected pipelines continue.
Model caches live in `.model_cache/`. Runs are serialized. Matching preserves native RGB for models that expect it and explicitly derives
grayscale for grayscale models; learned dense models retain their internal resolution rules.
The shared keypoint limit applies to sparse extraction and RoMa sampling, not LoFTR.
Results include panorama downloads, stage timings, pairwise match/inlier counts,
and JSON reports with settings, seed, device, and core package versions.
Timing includes initialization/downloads, so rerun after caches are warm for
useful speed comparisons. Inliers use symmetric transfer error <4 px; RANSAC
uses 3 px. These are consistency metrics, not accuracy scores. Visually inspect
outputs, especially on repetitive microscope textures.

MASt3R is an optional research adapter. Follow the
[upstream recursive installation](https://github.com/naver/mast3r), including its
DUSt3R dependencies, then start this server with the MASt3R repository on
`PYTHONPATH`. Its checkpoint is large and its CPU path has not been validated here.
Its noncommercial and training-dataset restrictions are shown separately.
Source/license references for all engines appear in the UI; see
[LightGlue](https://github.com/cvg/LightGlue),
[XFeat](https://github.com/verlab/accelerated_features),
[LoFTR](https://github.com/zju3dv/LoFTR), and
[RoMa](https://github.com/Parskatt/RoMa). Metadata is not a blanket clearance of
all checkpoint or training-data rights.

Comparison API: `GET /api/engines`, `GET /api/demos`,
`POST /api/comparisons` (multipart `options` JSON plus optional `files`),
`GET /api/comparisons/{id}`, and `GET /api/comparisons/{id}/report`.
Reports and images remain in `uploads/comparisons/`; live job polling is in-memory
and does not resume across server restarts. The app is intended for local use.

Validation: install `httpx` and run
`.venv/bin/python -m unittest discover -s tests -v`.

---

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
| `--alignment` | `homography` | `translation` restricts alignment to x/y shifts (same scale and orientation); skips perspective ECC refinement |
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

## Guided stitching

Open `/guided` (Grow a mosaic on the overview). Upload 2–60 images, select a
main image, then click remaining thumbnails to add them one at a time. Successful
additions gain valid pixel coverage and mark the image as stitched. Failed or
non-extending additions leave both the mosaic and used-image list intact; retry
after adding a connecting image. Already-used images cannot be added twice.

This tool exclusively uses [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR)
(full outdoor pretrained model), with no fallback matcher. Other matcher requests
are rejected by the API. The model matches each new image against original placed
images, followed
by translation-only displacement consensus and image blending. No homography,
affine transform, rotation, scale, shear, or perspective is estimated. Original
images are placed with whole-pixel x/y offsets without resampling. Images must
have the same scale and orientation. Multiple strong placement hypotheses must
agree. The canvas is capped at 40 megapixels / 16,000 pixels per side. Output is
an 8-bit color PNG, with no auto-cropping or whole-canvas rescaling. Download at
any stage, view at actual size, or restart with a different main image using the
same uploads. Sessions survive page reloads but not server restarts. Each accepted
map version remains in `uploads/guided/` for inspection.


`./start` sets up Grow a mosaic automatically. To repeat its setup manually:

```bash
.venv/bin/python setup_efficientloftr.py
./start
```

Setup installs pinned inference dependencies, checks out upstream revision
`07e9c1401e71e9c556b1fda9d86d060af36ba176`, and downloads and checksum-verifies
the authors' outdoor checkpoint (~193 MB). It updates one obsolete Kornia import
in the upstream source for compatibility with Kornia 0.8.3. Source and weights
stay local in `vendor/EfficientLoFTR` and `weights/eloftr_outdoor.ckpt`.
Inference uses CPU or CUDA automatically, the upstream reparameterized full model,
and grayscale inputs capped at 832 pixels with dimensions divisible by 32.
Matched coordinates are mapped back independently along each axis to the original
image resolution before translation estimation. No downloads occur during stitching.

Grow a mosaic offers **Matching input → Edge detection (Canny)** before loading
an image set. It is off by default. With this option, both images are resized to
model resolution, lightly Gaussian-blurred, and filtered with Canny (50/150)
before EfficientLoFTR inference. Only matching uses the edges; saved mosaics and
thumbnails retain the original colors. The choice is fixed for that upload session
and survives page reloads and choosing a different main image. Reload the image
set to compare modes. Edges may help outlines but can remove useful texture;
they are not guaranteed to improve matching. API: `edge_filter=true` on
`POST /api/guided`; session responses report the active value.

## MatchAnything comparison option

The comparison page includes **MatchAnything · ELoFTR**, using the authors'
[cross-modality model](https://zju3dv.github.io/MatchAnything/), separate from
ordinary LoFTR and the EfficientLoFTR outdoor model used by Grow a mosaic.

```bash
.venv/bin/python setup_matchanything.py
./start
```

Select MatchAnything on `/compare`, select any other pipelines to compare, and
run the same images through them. CPU and CUDA are supported. Setup downloads
source from the authors' Hugging Face Space pinned to revision
`6a7bcb589ec8da3a9e861e799122beaa5eba2193` and extracts only the ELoFTR checkpoint
from their official weight archive (~483 MB download). It verifies the checkpoint
SHA-256 before installation. Two import compatibility patches keep modern Kornia
and the project's module namespace working. Inference uses the upstream ELoFTR
configuration, grayscale resizing, square padding with validity masks, and returns
matches in original image coordinates. No model downloads occur during comparison.
This option does not include the separate MatchAnything RoMa variant.

The classic workspace and pipeline comparison both offer **Alignment → Translation only (x/y shifts)**. This works for sequential and unordered batch stitching and for the classic incremental map. Unsupported overlaps fail without falling back to perspective alignment. The classic workspace and its batch/incremental APIs default to translation only; perspective (homography) remains selectable. Pipeline comparison and the CLI keep their existing defaults. From the CLI, use `python run_stitcher.py --input img1.jpg img2.jpg --engine sift --alignment translation -o panorama.jpg`.

In the **classic workspace**, tile filenames must include their logical row and
column. The default template `prefix_{rNN}_{cNN}.png` recognizes names such as
`tile_r00_c09.png` (row 0, column 9). `prefix` accepts any prefix, and `NN` accepts
one or more digits. Change **Filename template** for other naming schemes:
`scan_row{row}_col{col}.tif`, for example, matches `scan_row2_col17.tif`.
The preview displays detected positions and sorts them numerically. Batch
stitching searches horizontal and vertical grid neighbors, using image content
to estimate their pixel offsets; row/column indices alone do not specify tile
spacing. Missing or duplicate positions trigger a popup and block the run until
corrected. Incremental tiles must have an unused position adjacent to the map;
the template is fixed until that map is reset. The classic API also validates
names (`filename_template` on `/api/stitch` and `/api/incremental/start`).
For random images without row/column names, use the separate **Pipeline comparison**
workspace at `/compare` and select **Unordered · find overlaps**.

For two-stage batch stitching, select **Batch stitching mode → Rows first, then
combine** (`stitching_mode=rows_first` on `/api/stitch`). Tiles are sorted by
column within each row and stitched into intermediate row images. Both stages
use mask-aware feather blending and the selected alignment and matcher. After
processing all rows, the second stage searches the completed rows for reliable
overlaps and combines the largest connected group. A failed row does not stop
the remaining rows. If none of the completed rows connect, the largest completed
row becomes the result. Partial results list omitted rows and count only included
tiles. Every completed row has its own download, retained when saving a project.
The progress timeline shows each completed row, and the mode is saved with
projects. Adjacent tiles and row images must overlap. Intermediate rows retain their original pixel scale
and coverage; the output size limit and auto-crop apply to the final panorama.
The default batch mode is **Rows first, then combine**; **Grid neighbors (single pass)** remains selectable.

To debug a classic grid run, click a tile in the floating grid or choose
**Inspect overlaps** below the upload area. Orange borders indicate rejected
neighbor overlaps; red borders indicate rejected pairs whose images belong to
different connected groups. The inspector lists the selected tile's tested
neighbors, with rejected pairs first, and shows filenames, rejection reasons,
match/inlier counts, and side-by-side matching previews. Lines show candidate
matches colored by relative matcher confidence, **not** geometric inlier status;
at most 500 lines are drawn per pair. Zero-match pairs still show both images.
Diagnostics remain available after a failed run and their previews are included
when saving the session as a project. Runs made before this feature must be
rerun to record the previews.
