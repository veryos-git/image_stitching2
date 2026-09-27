# Implement learned matching algorithms in `/classic`

Implementation handoff — 2026-09-27. This document specifies proposed work; it does not imply the algorithms are already implemented or tested. The [research report](ML_FEATURE_MATCHING_RESEARCH.md) contains the full inventory, upstream links, and licensing findings.

## Implementation record

The shared catalog, Stage 1 integration, adapter contract, explicit inference verification,
classic controls, pair retries and durable diagnostics are implemented. Research entries
in Stages 2–3 remain explicitly blocked until their adapter/checkpoint/runtime work is
complete. See [implementation and measured validation](validation/classic_ml/README.md)
for the exact scope, CPU evidence, unresolved work and 77-tile results. The acceptance
checklist below remains the full target, not a claim that every research adapter is finished.

## Objective and existing behavior to preserve

Allow users to select and compare all viable learned matching pipelines from the research inventory directly in `/classic`. Integrate in stages, keeping unfinished methods visible as unavailable rather than runnable. Components such as descriptors and refiners must be assembled into validated pipelines before becoming selectable engines.

- Keep **Translation only** and **Perspective / homography** independent of the matching algorithm. Every supported engine must supply correspondences to the chosen geometric estimator.
- Keep filename positioning with default `prefix_{rNN}_{cNN}.png`, including names such as `tile_r00_c09.png`.
- If positions cannot be parsed, show the existing popup explaining row/column naming, the adjustable template, and the separate workflow for images without grid positions. Do not silently infer an unordered grid.
- Keep thumbnails in the detected row/column positions, including empty cells, inside the movable/resizable overlay with the entire grid fitted into view.
- Preserve neighbor-only grid matching, disconnected-component reporting, highlighted failed links/tiles, clickable side-by-side previews, and saved-project diagnostics.
- Retain existing SIFT/ORB and SuperPoint + SuperGlue as compatibility baselines; the new implementation scope is learned alternatives.

## Repository map

| File | Work |
|---|---|
| `stitcher/catalog.py` | Shared engine definitions, capabilities, configuration schema, installation/weight status |
| `stitcher/matching.py` | `build_engine()` dispatch and common adapter contract |
| `stitcher/alternatives.py` | Reuse current optional adapters; split into focused modules as the catalog grows |
| `stitcher/efficientloftr.py`, `stitcher/matchanything.py` | Normalize prepared-image and diagnostic output to the shared contract |
| `stitcher/stitcher.py` | Extend `StitchConfig`; keep model selection separate from geometry and blending |
| `stitcher/unordered.py`, `stitcher/grid.py`, `stitcher/translation.py` | Audit grid matching, diagnostic events, and geometric verification for all adapters |
| `server.py` | Expose shared catalog; validate engine options; persist resolved configuration and pair results |
| `static/index.html`, `static/app.js`, `static/style.css` | Dynamic selector, capability-driven controls, readiness messages, pair comparison UI |
| `comparison.py`, `static/compare.*` | Reuse shared catalog/options rather than maintain a second model definition |
| `tests/`, optional requirements/setup scripts | Contract, integration, UI tests and pinned optional environments |

Current limitations: `/classic` hardcodes SuperGlue/SIFT/ORB. Several alternatives already have backend adapters. `/compare` accepts only 2–36 images. Dependency detection in the catalog is not proof of successful weight loading or inference.

## Algorithm implementation checklist

Engine IDs below are proposed unless explicitly identified as existing. Each family needs a pinned source revision, exact checkpoint, preprocessing definition, adapter, readiness check, and passing smoke test before becoming runnable.

### Stage 1 — expose and validate existing adapters

| Existing ID | UI label | Integration task |
|---|---|---|
| `aliked-lightglue` | ALIKED + LightGlue | Reuse extractor-specific matcher; verify RGB handling and diagnostics |
| `disk-lightglue` | DISK + LightGlue | Same contract; use DISK-specific matcher weights |
| `xfeat` | XFeat | Sparse features with descriptor assignment |
| `xfeat-star` | XFeat semi-dense | Pairwise correspondences; sampled-point preview |
| `xfeat-lighterglue` | XFeat + LighterGlue | Use compatible LighterGlue weights; preserve real scores when available |
| `loftr-outdoor`, `loftr-indoor` | LoFTR — Outdoor / Indoor | Handle resizing/padding and coordinate restoration |
| `efficientloftr` | EfficientLoFTR | Existing dedicated engine; add catalog entry and normalize preview fields |
| `matchanything-eloftr` | MatchAnything — ELoFTR | Normalize dedicated adapter and persist checkpoint identity |
| `tiny-roma`, `roma` | TinyRoMa / RoMa | Sample dense matches; map normalized coordinates correctly |
| `mast3r` | MASt3R | Optional adapter exists; finish dependency/setup, runtime and license status |

`roma` means the existing original RoMa adapter, not RoMa v2. Readiness must be rechecked in the actual runtime environment.

### Stage 2 — add the strongest new candidates

| Proposed ID/family | Pipeline requirement |
|---|---|
| `loma` | Expose verified B/B128/L/G/R checkpoints as supported variants, not invented combinations |
| `rdd` | Expose supported sparse/semi-dense modes and compatible learned matching; use corrected 2026 checkpoints |
| `dedode` | Specify detector and descriptor versions; offer only verified assignment/matcher pairings |
| `dad-*` | Detector component: register complete pipelines only after selecting compatible descriptor and matcher |
| `roma-v2` | Dedicated adapter and supported model presets; preserve overlap/precision information |
| `ufm` | Released variants only; memory/device checks; do not advertise unreleased Tiny weights as ready |
| `edm-efficient` | Efficient Deep Feature Matching; keep distinct from equirectangular EDM |
| `matchanything-roma` | Separate backbone/checkpoint from MatchAnything-ELoFTR |

### Stage 3 — complete the experimental inventory

The research report links each family. Integrate each row as separate validated entries or explicit component combinations; grouping here does not imply one adapter fits all models.

| Group | Algorithms to account for | Work required |
|---|---|---|
| Other sparse features | ALIKE, RIPE, RIPE++, LiftFeat, SiLK, R2D2, Fast-R2D2, D2-Net, ASLFeat, DALF, ZippyPoint, SFD2, CLIDD, LF-Net, LIFT | Extractor + supported assignment; isolate legacy dependencies; unresolved releases stay unavailable |
| Point/line features | UPAL | Start with its point correspondences; line constraints require an explicit additional geometry contract |
| Other direct matchers | DKM, ASpanFormer, MatchFormer, TopicFM, TopicFM+, SE2-LoFTR, CasMTR, PATS, ASTR | Pairwise adapter with original-pixel coordinate output |
| Dense correspondence family | GLU-Net, GOCor-based pipelines, PDC-Net, PDC-Net+ | Assemble complete upstream configurations; retain uncertainty and overlap filtering |
| Generalization / multimodal variants | MINIMA, GIM, XoFTR, ReDFeat | Register concrete backbone/checkpoint combinations; label any SuperPoint dependency |
| Learned descriptor components | HardNet, HardNet++, SOSNet, ContextDesc, MATCHA | Pair with a compatible detector and assignment method; no descriptor-only engine entries |
| Hybrid detector | Key.Net | Register complete detector/descriptor pipelines; label hybrid status accurately |
| Refinement components | Patch2Pix, Keypt2Subpx, rotation steerers, affine steerers | Optional compatible stages; no arbitrary mixing with unsupported descriptors |
| Optical flow | RAFT, GMFlow, UniMatch | Experimental correspondence adapters; handle occlusion/nonoverlap and fit selected global geometry |
| Projection-specific | Equirectangular EDM | Separate specialist entry; disclose projection assumptions |
| SuperPoint-dependent comparisons | OmniGlue, SuperPoint + LightGlue, SuperGlue; inspect GlueStick and MINIMA/GIM variants | Label dependency explicitly; exclude from “Without SuperPoint” filter when applicable |

Coverage means every researched family has an implementation or a recorded blocking reason. It does not mean presenting an unverified research repository as a working option. Do not silently substitute a different model.

## Shared catalog and configuration

Extend the current catalog rather than duplicate model lists in JavaScript. Serve it through a shared endpoint, for example `GET /api/engines` (proposed route).

Each entry should contain:

- Stable `id`, display label, family, description, source URL, source revision, checkpoint ID/hash.
- `implementation_status`: planned / implemented / verified / blocked, with a concrete reason where relevant.
- Separate dependency, weight, and device status. “Dependencies present” must not be called “Ready”. A successful recorded smoke test identifies its environment and checkpoint.
- Capabilities: sparse or pairwise, grayscale/RGB, supported devices, reusable image features, real confidence availability, available variants/refiners, and whether SuperPoint is used at inference. Record training-teacher dependency separately.
- Typed option schema with defaults, bounds, allowed enum values, and UI help. Separate code/checkpoint/dependency license metadata and unresolved terms.

Store engine-specific settings in a versioned `engine_options` object. Keep alignment, geometric residual threshold, grid template, and blending as shared settings. Validate on the server; reject unknown engines, invalid values, unsupported combinations, and unavailable checkpoints before expensive work.

Migrate old saved configurations without breaking `superglue_weights`, existing engine IDs, or historical projects. Save the resolved options after applying defaults/presets, not only the values submitted by the browser.

## Common adapter contract

Use a common result type while retaining a compatibility bridge for current `extract(gray)` / `match(a, b)` adapters. The following is a proposed interface, not current API:

```python
prepared = engine.prepare(image, image_id)  # original size + preprocessing metadata
matches = engine.match_pair(prepared_a, prepared_b)
# matches.points0: float32[N, 2], original-image (x, y) pixels
# matches.points1: float32[N, 2], corresponding original-image pixels
# matches.scores: optional float32[N]; documented meaning, not universal probability
# matches.metadata: checkpoint, preprocessing, timings, sampling, diagnostics
```

- Geometry code consumes only correspondences; prepared model payloads remain private to the adapter.
- Sparse engines may cache per-image features. Pairwise/dense engines need not fabricate keypoints or descriptors just to satisfy preview code.
- Preserve each image's independent scale, crop and padding transform. Test different dimensions, non-square images, and padding to stride multiples.
- Retain native RGB when expected; derive grayscale explicitly where required. Do not silently reduce RGB models to repeated grayscale channels.
- Return valid empty arrays for no matches. Filter nonfinite/out-of-bounds coordinates and validate equal lengths.
- Use `None`/unavailable for missing confidence instead of displaying manufactured all-ones scores as certainty.
- Sample dense correspondences with spatial coverage and overlap checks; store the sampling policy/seed.
- Keep subpixel coordinates. Translation-only means one global x/y shift, not rounded matches or a dense deformation field.
- Load optional dependencies lazily; avoid downloads during catalog/page loading. Track download/load progress on explicit setup or model use, with actionable errors.
- Cache using model revision/options, preprocessing, image identity and device. Bound GPU memory and serialize unsafe model access. Use worker processes for incompatible dependency stacks when needed.

## `/classic` UI behavior

Use a searchable selector grouped into learned sparse, direct/dense, experimental and legacy options. Offer “Without SuperPoint” and “Installed” filters. Show unavailable algorithms with status and setup details. Selecting a planned entry must never start a job.

Render settings from model capabilities. Show SuperGlue weights and Sinkhorn iterations only for SuperGlue. Show checkpoint/variant controls where supported. Do not reuse one model's match-confidence threshold for another model without an explicit mapping. Make feature budget and dense sample budget distinct settings. Adapt Fast/Balanced/Best presets per engine and show the effective resolution/budget.

Keep model selection beside the independent alignment selector. Persist both with the project. Show model/checkpoint, runtime/device, correspondence count, geometric inlier count, and rejection reason in results.

The pair inspector should offer **Retry this pair with…** and **Compare selected models**. These actions produce separate diagnostic results and must not silently replace accepted mosaic constraints. Applying a chosen model to the full grid requires a normal rerun.

## Debugging and persistence

Save diagnostics incrementally so they survive a failed/disconnected run:

- Pair image IDs/names, row/column positions, engine/checkpoint/options and preprocessing.
- Candidate correspondences with scores when available, geometric inlier mask, estimated shift/homography, residuals, spatial coverage, elapsed time and structured rejection reason.
- Component membership and rejected edges crossing components; distinguish an unmatched neighbor from an image that failed to load.
- Side-by-side visualization with toggles for all matches, inliers, outliers and confidence. Show “no matches” without breaking the preview.
- Full correspondence data in bounded sidecar artifacts; send only a sampled visualization through progress events. Store transform coordinates in original-image space.

Use the existing grid overlay to highlight failures. Clicking a tile must enumerate the specific neighbor pairs involved. Old projects without diagnostics should explain that a rerun is needed.

## Verification and acceptance checklist

- [ ] Shared catalog drives `/classic` and `/compare`; no duplicated hardcoded learned-model lists.
- [ ] Every runnable engine passes a real inference smoke test for its supported device/checkpoint. Missing optional packages do not prevent server startup.
- [ ] Adapter tests cover unequal image sizes, RGB/grayscale, resize/padding inversion, no matches, invalid values and score absence.
- [ ] Known translated image pairs recover x/y within a documented tolerance. Geometric tests reject inconsistent correspondences. Negative-pair behavior is measured rather than assumed.
- [ ] Translation mode never introduces rotation, scale, perspective or dense warping. Homography remains independently selectable.
- [ ] Failure during model loading, matching or graph alignment preserves completed diagnostics and produces actionable UI errors.
- [ ] Filename parsing warnings, spatial grid layout, fit-to-overlay behavior, dragging/resizing and pair clicks still work.
- [ ] Save/reload preserves exact model configuration and failed-run pair results; old projects remain readable.
- [ ] CPU-only environments behave clearly; unavailable CUDA and GPU out-of-memory do not trigger an undisclosed model/device fallback.
- [ ] Compare the 22 pairs across rows 03/04 and 05/06 from the problematic dataset, plus successful neighbors and nonoverlapping negatives. Measure correct recovered edges, false bridges, residuals, runtime and memory.
- [ ] Run candidate winners on all 77 tiles / 136 intended neighbor pairs. Check graph connectivity and consistency, not just match count. Record failures honestly; connection is not guaranteed by integration.

For each algorithm, the completion record should include engine ID, source/checkpoint pin, setup instructions, tested device, smoke-test result, pair-debug screenshot/artifact, and known limitations.

## Dependency and licensing notes

Pin versions and checkpoints; separate optional dependency groups from the base application. A common experimental wrapper such as vismatch may reduce adapter work, but still requires coordinate/diagnostic validation and original model terms.

Carry forward the [research report's licensing findings](ML_FEATURE_MATCHING_RESEARCH.md#licensing-findings-relevant-to-selecting-implementations). In particular, inspect the exact vendored EfficientLoFTR/MatchAnything revisions rather than copying stale Apache labels; current upstream and historical source terms can differ. Track MASt3R/R2D2 restrictions and unresolved licenses explicitly. This is implementation metadata, not a claim that every listed model is cleared for every use.

Start with Stage 1 and the shared contract/diagnostics. Proceed through Stages 2 and 3 without weakening geometric validation to make a model appear successful.
