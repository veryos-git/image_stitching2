# Classic learned matching implementation and validation

2026-09-27. This record distinguishes implemented integration from unimplemented research families.

## Implemented

- One catalog and versioned, typed option schema drive classic and comparison pages. It contains all research families, explicit component blockers, capabilities, source/checkpoint metadata and setup instructions. Unknown/unsupported options and unavailable engines are rejected before a web job is allocated.
- Existing Stage 1 adapters are integrated into batch grids. Native RGB is preserved for ALIKED/DISK, RoMa/TinyRoMa and MASt3R; grayscale is explicit elsewhere. Original pixel coordinates use independent x/y scale factors. MatchAnything pads each image to its own square. Pairwise preparation has no fabricated descriptor/keypoint requirements. LighterGlue retains its native scores; adapters without confidence return `None`.
- Baseline incremental feature merging is preserved. Learned incremental matching uses original placed neighbor payloads, composes their transforms into map coordinates, rejects disagreeing translation proposals, and commits state only after successful placement. Geometry remains independent of model choice. Classic translations retain fractional offsets; guided stitching retains its documented integer-placement behavior.
- Grid runs save bounded correspondence sidecars (NPZ), candidate/accepted transforms, masks, residuals, coverage, timings, device, options and preprocessing. Rejected edges and matching exceptions remain inspectable. A maximum of 16,384 spatially sampled correspondences is retained per pair; each progress preview draws at most 500 lines. Sampling is deterministic 8×8 source-cell round-robin, ordered by native score where available. Geometry uses the same sampled set saved to the artifact.
- Inspector filters show all/inlier/outlier/confidence previews. Retry/compare actions use saved original inputs, serialize model computation, write separate artifacts and never replace the original graph. Applying a model to the mosaic requires a normal rerun.
- Saving projects copies diagnostics and rewrites artifact links. Reload restores resolved options, alignment, input files and spatial grid positions. Legacy configurations migrate without requiring model installation; old projects without diagnostics explain the need for a rerun.
- Explicit model verification records a known-shift unequal-size pair, an unrelated negative, coordinate error, loaded tensor hash, environment/package information and an image artifact. Learned alternatives remain unavailable until verification passes locally. Compatibility baselines retain their previous startup behavior. CUDA requests are rejected when CUDA is unavailable; a CPU verification does not authorize an untested CUDA path.

## Test evidence

The final CPU smoke sweep in `smoke-final.json` passed for all 15 tested engines:
SIFT, ORB, SuperGlue, XFeat, XFeat semi-dense, XFeat + LighterGlue,
ALIKED/DISK/SIFT + LightGlue, indoor/outdoor LoFTR, EfficientLoFTR,
MatchAnything-ELoFTR, TinyRoMa and RoMa. Each known-shift error is below 2 pixels;
none accepted the unrelated smoke-test negative. This is limited smoke evidence,
not a dataset benchmark or proof that every unsupported input/device will work.
The JSON includes loaded tensor hashes, pinned source/package information,
fixture identity and per-model timing. Local preview artifacts are served from
the corresponding `/api/engines/<id>/verification-preview` route.

All **60 regression tests pass**. The regression suite covers configuration rejection, original-coordinate and
RGB contracts, missing confidence, subpixel-only translation, failed-pair artifact
retention, original-neighbor incremental matching, state preservation, retries,
legacy configuration migration and project persistence. Chromium checks cover
catalog filters, generated controls, unavailable research entries, a successful
grid job, inspector filters/retry, project save/reload, comparison controls and
filename warnings. `pair-inspector.png` captures the tested inspector.

## Verification commands

```bash
./start
.venv/bin/python verify_engines.py xfeat aliked-lightglue efficientloftr --device cpu
.venv/bin/python -m unittest discover -s tests -v
# With an isolated server on port 8769 and Playwright/Chromium installed:
PLAYWRIGHT_BROWSERS_PATH=/tmp/stitch-playwright .venv/bin/python tests/browser_classic.py
.venv/bin/python evaluate_classic.py testdata/scan_2026-09-27_175604_6149a233/dowscaled
.venv/bin/python evaluate_classic.py testdata/scan_2026-09-27_175604_6149a233/dowscaled --full
```

Use **Verify model inference** in either page for the same explicit verification workflow. Downloads and inference happen only on explicit verification/use, never while listing the catalog. Evidence is local to `.model_cache/verification`; changes to runtime/source/checkpoint files invalidate readiness. Checkpoint/dependency/download errors are shown instead of changing models or devices.

## Measured 77-tile results

CPU, translation only, matching resolution 512 px, default per-engine budgets, 3 px geometric residual threshold. Every full run tests 136 intended neighbor pairs plus three far-apart same-row expected negatives. Boundary screening tests the 22 requested links, six other neighbors and three expected negatives. Results are geometric consistency measurements, **not annotated registration accuracy**. No manual landmark ground truth is available. A far-apart grid pair is an expected negative, not independently labeled proof of nonoverlap.

| Engine | Accepted neighbors | Requested boundaries | Accepted negatives | Components | Largest group | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| aliked-lightglue | 26/136 | 0/22 | 0/3 | 51 | 5 | 20.9 |
| efficientloftr | 37/136 | 0/22 | 0/3 | 40 | 8 | 87.3 |
| sift | 15/136 | 0/22 | 0/3 | 62 | 8 | 2.2 |
| xfeat | 0/136 | 0/22 | 0/3 | 77 | 1 | 4.6 |
| xfeat-lighterglue | 27/136 | 0/22 | 0/3 | 50 | 10 | 52.1 |

None of these configurations connected the scan or recovered a requested boundary link. Geometric checks were not relaxed. Zero cycle discrepancy on a fragmented forest does not demonstrate global consistency: many components have no cycles. Higher input resolutions or different validated models may behave differently; the integration does not establish that any model fixes this dataset.

`grid-summary.json` contains aggregate measurements, including process peak RSS. Each multi-engine process reports its cumulative high-water RSS, not isolated model memory. Full per-pair images, correspondences, reports and rejected-link reasons remain in `uploads/classic-evaluation/{boundaries,full}/<engine>/`. `pair-inspector.png` is the browser-test artifact.

## Research inventory and remaining limits

Stage 2 and Stage 3 families are **accounted for, not implemented**: their catalog entries remain blocked with concrete missing prerequisites (e.g. corrected checkpoint pairing, legacy worker runtime, component composition, occlusion filtering or projection-specific geometry). MASt3R retains its existing lazy adapter and setup instructions but was not installed or verified here. There is no undisclosed substitute model. No GPU was available; CPU evidence does not validate CUDA memory or inference.

This delivery therefore completes the shared integration/diagnostic infrastructure and existing-adapter exposure, not every proposed research adapter. Adding any blocked family still requires its pinned source/checkpoint, supported dependency recipe, coordinate adapter and real local verification.

Source licenses were inspected locally: pinned EfficientLoFTR carries Project Registration License v1.0; the pinned MatchAnything Space source carries Apache-2.0. Checkpoint, dependency and dataset terms remain separate unresolved metadata. XFeat setup now pins commit `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`; LightGlue, Kornia, RoMa, EfficientLoFTR and MatchAnything use the pinned sources/releases recorded by the catalog and requirement/setup files.
