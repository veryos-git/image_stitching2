# External Scan workflow vs. `/classic`

Compared on 2026-09-28. The external system is described by [external_workflow.md](external_workflow.md), which reports inspection of revision `649180e`; its implementation and runtime observations were not independently reproduced here. The `/classic` side was checked against this repository's current source. This comparison covers actual web defaults, rather than the different defaults of the generic Python configuration or CLI. The separate `/microscope`, `/compare`, and guided workflows are outside scope.

## Main finding

Both workflows can build translation-only mosaics from row/column-named images, but their placement strategies differ substantially. The external Scan workflow registers neighboring tiles using normalized cross-correlation (NCC), optimizes their positions together, and composites original-resolution images. `/classic` defaults to learned correspondence matching, stitching each row independently, then matching and combining the completed row images. Its optional single-pass grid mode uses a spanning tree, not a global position optimization.

Consequently, selecting translation alignment and grid neighbors in `/classic` does **not** reproduce the external algorithm. The biggest differences are global consistency, handling of missing matches, retained image resolution, and photometric correction. Neither description establishes that one produces more accurate mosaics on the same data.

## Workflow comparison

| Aspect | External Map → Scan | This project's `/classic` |
| --- | --- | --- |
| Inputs | Integrated serpentine capture; original PNGs plus copies at most 256 px wide. | Upload existing images; batch and incremental modes. No integrated motor/camera scan in this route. |
| Default pipeline | One tile graph, global layout, correction, compositing. | DISK + LightGlue; translation; rows first, then combine. |
| Grid interpretation | Infers complete rectangular grids with at least four recognized tiles; otherwise falls back to all pairs. | Requires every uploaded filename to match a configurable row/column template and have unique coordinates. No unnamed-input fallback. A complete rectangle is **not** required by filename validation. |
| Candidate pairs | Immediate right/down neighbors for recognized grids; no diagonals. | Default: consecutive column-sorted tiles within each row, then all pairs of completed rows. Optional grid mode: tiles whose numeric Manhattan grid distance is exactly one. |
| Registration | High-pass grayscale NCC using FFT/integral-image statistics; local refinement and phase-correlation residual. | Selected engine produces point correspondences; geometric consensus estimates translation or homography. SIFT, ORB, SuperGlue and multiple learned alternatives are available subject to readiness checks. |
| Resolution | Registration at ≤256 px width; shifts converted to original-image pixels; originals used for output. | Inputs default to ≤1080 px width **before both matching and blending**. A separate model matching-size limit can reduce them further for feature inference, with coordinates mapped back to the processed input. Set input width to `0` to preserve original tile resolution. |
| Layout | Weighted least squares with six Huber reweighting iterations over graph constraints. | Within rows: chained pair transforms. Between rows, or in grid mode: maximum-inlier spanning tree. No joint adjustment using all accepted loop constraints. |
| Failed matches | Low-weight grid predictions, layout-guided NCC retry, optional LoFTR proposals verified with NCC. | No equivalent automatic grid prediction or layout-guided rescue. Inspector retries compare models separately; a new stitch run is needed to apply a model change. |
| Disconnected data | Can emit warnings and still composite components whose relative placement is unresolved. | Grid mode rejects disconnected inputs. Rows-first may omit failed/disconnected rows and return a marked partial result; if row combination fails, returns the completed row with greatest covered area. |
| Illumination | Estimated per-pixel/color flat field, plus scalar exposure gains; capture correction may already be baked in. | Scalar exposure compensation; no equivalent flat-field estimation in the classic pipeline. Optional Sobel preprocessing changes the output itself into an edge mosaic. |
| Blending | Weighted averaging; the supplied document reports uniform weights for the default feather mask in its tested OpenCV environment. | Rows-first explicitly uses mask-aware feather blending in both stages. Single-pass grid mode uses multiband/Laplacian blending, six levels by default. |
| Output | 8-bit color PNG; optional JPEG preview; positions and pair-report JSON. | Final JPEG at quality 95; intermediate row downloads, progress events, statistics, and grid diagnostics. Projects retain inputs, settings, events and artifacts. |
| Execution | Durable FIFO queue, one stitch process, two registration workers; restart recovery. | Background job threads serialize stitching through a shared compute lock. Active jobs/events are in memory; explicitly saved projects persist. No equivalent durable restart/resume queue. |

Sources: external description §§2–9; [web/API defaults and jobs](server.py), [input preprocessing](stitcher/preprocessing.py), [filename validation](stitcher/grid.py), [rows-first execution](stitcher/rows.py), [graph alignment](stitcher/unordered.py), and [compositing](stitcher/blender.py).

## Differences that affect results

### Match acceptance and alignment accuracy

External NCC defaults to a correlation threshold of `0.3`, with overlap-area/variance checks. Optional LoFTR is a rescue mechanism, not its primary matcher, and its proposals must still pass NCC.

Classic translation fitting uses deterministic displacement consensus and median refinement. It requires at least 12 inliers, at least 50% of correspondences, inlier convex-hull coverage of at least 1% of each image, and at least 3% overlap of the smaller image. Its default geometric tolerance is **50 processed-input pixels**. This tolerance and the external NCC score measure different things and cannot be compared numerically. Classic's actual alignment path retains fractional translations and applies interpolated warping; the helper's default integer-rounding option is overridden. See [translation fitting](stitcher/translation.py) and [its caller](stitcher/unordered.py).

The external system preserves original image detail in compositing, but has no original-resolution registration refinement. Classic can also infer matches on reduced images. Neither original-resolution output nor fractional offsets prove original-pixel registration accuracy.

### Global consistency and partial results

The external solve can distribute inconsistent edge measurements across a connected graph and downweight outliers. Classic's tree placement uses selected edges and does not reconcile remaining loops. Rows-first additionally freezes each row into a raster before matching rows, so cross-row evidence cannot revise individual tile positions inside an already completed row.

External predicted edges can bridge weak-texture areas, but these are inferred placements. Classic's grid mode requires measured connectivity; its default rows-first mode instead permits explicitly reported omissions. A successful output therefore has different meanings in the two systems: inspect external warnings/priors and classic `partial`, `included_rows`, `excluded_rows`, and `failed_rows` before interpreting the mosaic as complete.

Grid labels also differ in meaning: the external inference remaps recognized labels to contiguous indices, while classic single-pass grid adjacency uses their numeric differences. Classic rows-first simply joins consecutive uploaded tiles after column sorting, even if column labels have gaps; those tiles still need actual visual overlap.

### Photometry and blending

External flat-field correction addresses spatial shading within each tile, which classic's single scalar gain cannot remove. External gains are clamped to `[0.75, 1.35]`; classic estimates log-domain gains from median overlap intensities and clamps them to `[0.4, 1.6]`. In rows-first, compensation can run once within rows and again between row images.

Classic's batch feather implementation pads masks with a zero border, providing varying edge weights even when a valid image fills the canvas. This differs from the external default-mask behavior reported in the supplied document. Classic single-pass multiband blending is another substantive difference; neither blending method repairs incorrect registration. See [batch blend dispatch](stitcher/stitcher.py) and [mask/gain implementations](stitcher/blender.py).

### Memory, output size, and cropping

External compositing uses horizontal bands, but retains the complete output and a tile cache in memory; its output-size limit is applied **after** full compositing. Classic holds each warped image and mask on the stage's canvas, with additional blending buffers. Its normal 12,000 px longest-side guard reduces the canvas **before** warping. However, intermediate rows explicitly disable that guard to preserve scale, so large row stages can still be expensive.

Classic auto-crops by default, which can remove covered peripheral content to obtain an interior rectangle. Intermediate rows only trim empty outer padding. External Scan preserves the mosaic canvas, including black uncovered regions. Neither workflow is an out-of-core, multiresolution scientific-image export pipeline in the paths compared here.

## Diagnostics and evidence limits

Classic single-pass grid runs provide saved correspondences, geometric masks, residuals, rejection reasons and graph information, plus pair inspection and model comparison. Rows-first child runs explicitly disable persistent pair-diagnostic directories; their progress images and row statistics should not be mistaken for the same saved tile-neighbor diagnostic coverage. See [diagnostics](stitcher/diagnostics.py) and [row child configuration](stitcher/rows.py).

The external report provides NCC, displacement, overlap, edge kind and residual, but its description says synthetic prior constraints are not fully exported. Its positions file uses first-tile-relative original pixels and omits explicit canvas origin/output scale. Classic project grid positions are logical row/column metadata, not an equivalent standalone original-pixel positions export.

Existing [classic validation](validation/classic_ml/README.md) records a 77-tile experiment where five tested configurations failed to connect the scan. Those runs used 512 px matching and a 3 px tolerance, so they do not characterize the current 50 px UI default or prove comparative performance against the external stitcher. The external document reports two focused tests, not a matched real-slide benchmark. No new stitching, inference, or performance tests were run for this documentation comparison.

## Implications for future work

To bring classic closer to the external design, the substantive additions would be an NCC registration option, robust global translation optimization, layout-guided retries with explicitly labeled priors, and optional flat-field correction. Keeping separate registration copies and original compositing images would also avoid default input resizing reducing output detail. Band/tile compositing and durable job state would address separate scaling and operational differences.

A useful next comparison would run both systems on identical original tiles and score landmark error, retained tile coverage, visible seams, runtime and peak memory. Count predicted placements and partial outputs separately; do not rank systems solely by whether they emitted an image.
