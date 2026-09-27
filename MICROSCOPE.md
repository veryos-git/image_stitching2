# Microscope mosaicing

Implementation follows `microscope_mosaicing_implementation_plan.md` (the file
present in this checkout). This is a separate CPU-first engine and script;
it does not reuse the panorama homography pipeline or recursively stitch mosaics.

Open **http://localhost:8001/microscope**, also linked from the comparison page.
Choose a 2×2, 4×5, or full demo grid, or upload a schema-1.0 manifest and originals.
Row/column must be spatial cells, including for serpentine acquisitions. Uploaded
manifest paths are matched to uploaded filenames; the server does not read arbitrary
client-specified local paths. Inputs are never overwritten.

```bash
.venv/bin/pip install -r requirements-microscope.txt
.venv/bin/python microscope_mosaic.py demo-manifest \
  --demo-dir demo_content/scan_2026-09-17_010421 --out /tmp/slide.json
.venv/bin/python microscope_mosaic.py validate --manifest /tmp/slide.json --out /tmp/validated.json
.venv/bin/python microscope_mosaic.py register --manifest /tmp/slide.json --out uploads/my-coarse
.venv/bin/python microscope_mosaic.py render --run uploads/my-coarse --refiner identity --out uploads/my-baseline
.venv/bin/python microscope_mosaic.py refine --run uploads/my-coarse --refiner pixelstitch --out uploads/my-pixelstitch
.venv/bin/python microscope_mosaic.py benchmark --run uploads/my-coarse --out uploads/my-comparison
```

`run --manifest ... --out ...` combines registration and export. `--config` accepts
JSON or YAML; see `microscopy.pipeline.DEFAULTS` for the implemented schema.
`coarse.primary` and `coarse.fallback` select `phase_correlation` or
`feature_translation`. `refinement.backend` selects `identity`, `farneback`, or
`pixelstitch`. No model imports occur in the identity baseline.

For complete-pipeline benchmarks, use `benchmark --dataset slides.json --matrix
matrix.yaml --out ...`. Dataset format: `{"slides":[{"manifest":"slide.json",
"split":"development"}]}`. Matrix format maps pipeline names to configuration
objects. All attempted slides, failures, declared splits and optional landmark
metrics are retained. For refinement comparisons, `--run` holds coarse poses fixed.
These commands do not automatically establish scientific acceptance or create a
locked dataset from one slide.

Implemented behavior:

- Manifest uniqueness/dimension/dtype/channel/plane checks, input hashes, optional
  explicit masks; original uint8/uint16/float32 YX/YXC data remain untouched.
- Right/bottom neighbor graph. Grid topology is not substituted for measured offset.
- Bounded linear phase search with multiple hypotheses, SIFT translation fallback,
  full-resolution overlap validation, and graph-informed retry. Physically plausible
  cross-component candidates may be added when several verified edges establish
  both grid strides. No measured spacing means no invented distant edges.
- Component-anchored global translation solve with Huber loss and uncertainty-aware
  weights where available. Inconsistent loops are flagged for review, not silently
  treated as reliable. Position uncertainty is explicitly unavailable.
- Disconnected components and isolated tiles exported separately. Singletons remain
  `unresolved`; a set with no usable pair is `failed` even though source previews
  can be inspected. A partial render never becomes a complete slide.
- Fixed coarse poses, one regularized bilinear mesh per tile, all accepted pair
  constraints solved together, zero mean deformation, magnitude/strain/residual
  checks, contraction-based inverse mapping and group fallback to coarse geometry.
- Chunked, memory-mapped uncompressed BigTIFF debug export at original precision;
  explicit coverage count, valid-black support, deterministic feather weights,
  no exposure correction, original-source backward sampling. No source pyramid
  or scientific production pyramid is claimed. RAM scales with previews, one
  original tile, meshes and a render chunk rather than the full mosaic canvas.
- Atomic stage completion, reusable registration/refinement cache keys, original
  mutation checks, rerendering without rerunning registration. Incomplete TIFFs
  have `.partial.tif` names. Chunk-level resume is not implemented.
- Manifest, run/summary JSON, edge attempts, poses, maps, reconstruction provenance,
  residual CSV, overlap before/after/difference previews, QC HTML and TIFF artifacts.

PixelStitch uses the official source pinned in `microscopy/pixelstitch.lock.json`.
It runs in an isolated, timeout-bounded worker to avoid upstream `modules` imports
colliding with other engines. Install with:

```bash
git clone https://github.com/MakiseChris666/PixelStitch vendor/PixelStitch
git -C vendor/PixelStitch checkout 5893fde63ab41a97a20398b6745202ca97aa2b2c
```

Install matching torch/torchvision separately for this optional worker. The core
validates the checkpoint SHA-256. The worker replicates normalized grayscale into
three channels, initializes translations in a shared canvas, runs four inference
iterations, and converts both upstream backward sampling grids through that shared
frame into original-tile correspondences. It does not negate a forward field or
use the model's blended panorama as a tile. The engine replaces the upstream
homography resampler explicitly; this is an experimental integration, not a claim
of byte-for-byte upstream output parity.

Identity is the default. To *apply* a nonzero mesh, provide all of:

```yaml
refinement:
  backend: farneback
  acceptance_profile:
    max_displacement_px: 2.0
    max_strain: 0.05
    max_constraint_error_px: 1.0
```

Those values are an **illustrative development configuration**, not established
microscopy tolerances. Without limits, candidate refinements can be evaluated but
the coarse baseline is retained. Backend success, mesh acceptance, fallback and
scientific QC are separate report fields. Photometric consistency is only an
engineering gate. PixelStitch ran on demo overlaps during implementation and was
rejected for no photometric improvement; it is not microscopy-validated.

Remaining scientific/deployment gates from the plan: calibrated stage/manual-anchor
placement, calibrated uncertainty, locked real-slide datasets and acceptable
measurement distortion, physically calibrated deformation limits, production
pyramid/reader selection and representative hardware/resource budgets. The current
implementation does not apply external stage coordinates or manual anchors; it
keeps those placements unresolved. It has no affine mode or cross-channel registration.
`qc_accepted` intentionally remains false until these scientific gates are resolved.

Optional independent evaluation landmarks can be supplied on a manifest as
`evaluation_landmarks: [{tile_a, tile_b, point_a: [x,y], point_b: [x,y]}]`.
They never enter registration/refinement acceptance. Same-component errors report
count, median and p95 in original pixels. Cross-component landmarks are unscorable.

Run deterministic CPU tests:

```bash
.venv/bin/python -m unittest discover -s tests -p test_microscopy.py -v
```

Implementation validation notes (2026-09-24): the 2×2 demo produced two supported
row components; the 20-tile registration produced four row components. Vertical
connections were not verified and were not fabricated. Confirm the demo's spatial
row/column semantics before interpreting this as microscope stage geometry.
A custom uint16 upload completed through the browser with a uint16 TIFF export.
The upstream PixelStitch CLI smoke example required CPU checkpoint mapping and
then emitted invalid-value warnings on the synthetic pair; this is **not** counted
as a successful upstream output-parity test. The isolated adapter independently
rejects non-finite proposals and retains the coarse baseline when checks fail.
