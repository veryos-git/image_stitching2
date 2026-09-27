# Learned image matching alternatives for tile stitching

Research date: 27 September 2026. Scope: machine-learning approaches that can supply correspondences for image stitching. This is a broad research inventory, not a claim to enumerate every paper or checkpoint. Recommendations below are engineering judgments for this project, not benchmark results on the uploaded images. Links point to original implementations or author papers.

## What to try first

Start with **ALIKED + LightGlue, DISK + LightGlue, and XFeat + LighterGlue**. These replace SuperPoint, represent different learned feature extractors, and already have adapters in this repository. Then test a detector-free family: **LoFTR / EfficientLoFTR / MatchAnything-ELoFTR**. For a more expensive dense alternative, test **TinyRoMa, RoMa, and eventually RoMa v2**. Add **LoMa, RDD, and DeDoDe** as the next sparse research candidates.

SuperPoint detects points and computes descriptors. SuperGlue and LightGlue match features between images. Changing SuperGlue to LightGlue while retaining SuperPoint does not replace SuperPoint. LightGlue must use a checkpoint trained for its particular descriptor family; arbitrary feature/checkpoint combinations are invalid. Its released alternatives include ALIKED and DISK. [LightGlue implementation](https://github.com/cvg/LightGlue)

Translation-only stitching is compatible with sparse, semi-dense, and dense learned correspondences. Fit and validate a single x/y shift from those correspondences; do not use a network's unconstrained dense warp as the final mosaic geometry.

| First-round candidate | Why it belongs in the experiment | Practical expectation, not measured here |
|---|---|---|
| ALIKED + LightGlue | Lightweight learned detector/descriptor with a supported learned matcher | Good first general-purpose alternative |
| DISK + LightGlue | A different learned feature-selection objective | Useful independent comparison to ALIKED |
| XFeat + LighterGlue; XFeat sparse / XFeat* | Efficient features with sparse and semi-dense routes | Prioritize when CPU time matters |
| LoFTR indoor and outdoor | Matches images without first selecting a sparse keypoint set | Test both checkpoints; indoor is not a synonym for microscopy |
| EfficientLoFTR | Faster detector-free family | Already a separate engine here; requires adapter work for classic diagnostics |
| MatchAnything-ELoFTR | Training for broader appearance/modality variation | Test as a separate checkpoint family, not just another UI label |
| TinyRoMa / RoMa | Dense matching provides a substantially different failure profile | Tiny first in a CPU environment; larger model preferably on GPU |

Sources: [ALIKED](https://github.com/Shiaoming/ALIKED), [DISK](https://github.com/cvlab-epfl/disk), [XFeat and LighterGlue](https://github.com/verlab/accelerated_features), [LoFTR](https://github.com/zju3dv/LoFTR), [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR), [MatchAnything](https://github.com/zju3dv/MatchAnything), [RoMa and TinyRoMa](https://github.com/Parskatt/RoMa).

## What this application already contains

The classic page currently offers `superglue`, `sift`, and `orb`. Thus its only learned choice still uses SuperPoint. The comparison page has a substantially broader catalog.

| Local engine ID | Where it is wired | Audit result |
|---|---|---|
| `aliked-lightglue`, `disk-lightglue` | Comparison catalog and alternative adapter | Dependencies detected |
| `xfeat`, `xfeat-star`, `xfeat-lighterglue` | Comparison catalog and alternative adapter | Dependencies detected |
| `loftr-outdoor`, `loftr-indoor` | Comparison catalog and alternative adapter | Dependencies detected |
| `roma`, `tiny-roma` | Comparison catalog and alternative adapter | Dependencies detected; `roma` is not RoMa v2 |
| `matchanything-eloftr` | Comparison catalog and dedicated engine | Setup detected |
| `efficientloftr` | Dedicated engine / guided workflow | Not exposed in comparison catalog; feature data contract differs |
| `mast3r` | Comparison catalog and optional adapter | Dependencies not detected |

Dependency detection does not prove that every weight file has downloaded or that inference succeeds. This research did not benchmark or install these models. The inspected Python environment reports CUDA unavailable; GPU advice below assumes a separate CUDA-enabled environment.

Local evidence: `static/index.html`, `stitcher/catalog.py`, `stitcher/alternatives.py`, `stitcher/efficientloftr.py`, `stitcher/matchanything.py`, and `comparison.py`. The comparison uploader currently accepts **2–36 images**, so use selected pairs/subsets there; the 77-image run cannot be tested there in one upload yet.

## Learned sparse features and complete sparse pipelines

These are replacements for SuperPoint or building blocks for such replacements. A detector/descriptor still needs correspondence assignment: nearest-neighbor matching, a compatible trained matcher, or a pipeline supplied by its authors. Using a conventional nearest-neighbor assignment does not make the learned feature extractor classical.

| Method / family | What is learned | Relevance and maturity |
|---|---|---|
| [ALIKED](https://github.com/Shiaoming/ALIKED) | Detection and deformable descriptors | High priority; released LightGlue pairing |
| [ALIKE](https://github.com/Shiaoming/ALIKE) | Lightweight detection and descriptors | Earlier family; useful baseline, ALIKED first |
| [DISK](https://github.com/cvlab-epfl/disk) | Detector/descriptor trained for matching | High priority; released LightGlue pairing |
| [XFeat / XFeat* / LighterGlue](https://github.com/verlab/accelerated_features) | Efficient local features and optional learned assignment | High priority; sparse and semi-dense variants |
| [DeDoDe / DeDoDe v2](https://github.com/Parskatt/DeDoDe) | Decoupled detection and description | High-priority next addition; record exact detector/descriptor checkpoints |
| [DaD](https://github.com/parskatt/dad) | Diverse keypoint detection | Detector only; needs descriptor and matcher |
| [LoMa](https://github.com/davnords/LoMa) | Local features and learned matching | 2026 released family; B/B128 first, larger and rotation-oriented variants later |
| [RDD](https://github.com/xtcpete/rdd) | Deformable detector/descriptor; own LightGlue models | 2025 family with 2026 updates; test revised checkpoints |
| [RIPE](https://github.com/fraunhoferhhi/RIPE) | Local features trained from unlabeled image pairs | Interesting alternative to depth/pose-supervised features |
| [RIPE++](https://github.com/fraunhoferhhi/RIPEpp) | Joint feature and matcher training from positive pairs | Newer research candidate; verify exact inference setup |
| [LiftFeat](https://github.com/lyp-deeplearning/LiftFeat) | Local features trained with geometric cues | Relevant hypothesis for weak/repeated texture; not established on these tiles |
| [SiLK](https://github.com/facebookresearch/silk) | Self-supervised local keypoints/descriptors | Useful independent training approach |
| [R2D2 / Fast-R2D2](https://github.com/naver/r2d2) | Repeatability/reliability-aware local features | Established research baseline; noncommercial terms |
| [D2-Net](https://github.com/mihaidusmanu/d2-net) | Joint dense features and detection | Older, comparatively heavy baseline |
| [ASLFeat](https://github.com/lzx551402/ASLFeat) | Deformation-aware feature detection/description | Older TensorFlow integration cost |
| [DALF](https://github.com/verlab/DALF_CVPR_2023) | Deformation-aware local features | More relevant when deformation/rotation is present |
| [ZippyPoint](https://github.com/menelaoskanakis/ZippyPoint) | Efficient learned local features | Additional speed-oriented baseline |
| [SFD2](https://arxiv.org/abs/2304.14845) | Semantic-guided feature detection/description | Domain assumptions need checking for microscopic texture |
| [CLIDD](https://arxiv.org/abs/2601.09230) | Independent deformable description across layers | 2026 research candidate; paper verified, author release not fully audited |
| [UPAL](https://github.com/francois141/upal) | Unified point and line features | 2026 specialist option; teacher set includes SuperPoint, even though inference is a different model |
| [LF-Net](https://github.com/vcg-uvic/lf-net-release) | Learned local feature pipeline | Historical research baseline; legacy environment |
| [LIFT](https://github.com/cvlab-epfl/tf-lift) | Learned detection, orientation, description | Historical baseline; low priority for new integration |

LoMa combines ideas/components from DeDoDe, DaD, and LightGlue; it is not simply a new SuperPoint checkpoint. Its release includes several size variants and a rotation-oriented variant. This makes it a promising next integration, but its benchmark claims are not evidence of success on this project's tiles. [LoMa authors](https://www.davnords.com/loma/)

RDD's maintainers report that they retrained both RDD and LightGlue in May 2026 to correct data leakage. Record the revised checkpoint identifier and do not mix old benchmark numbers with new weights. [RDD release notes](https://github.com/xtcpete/rdd)

## Detector-free, semi-dense, and dense learned matching

These compare two images directly and return correspondences or a correspondence field. They avoid depending on independently detected SuperPoint keypoints. Dense predictions still need overlap/confidence filtering: pixels outside the shared field of view should not contribute invented constraints.

| Method / family | Main distinction | Priority for this application |
|---|---|---|
| [LoFTR](https://github.com/zju3dv/LoFTR) | Transformer coarse-to-fine matching | First round; indoor and outdoor checkpoints |
| [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) | Efficient semi-dense matching | First round, after adapter compatibility |
| [EDM: Efficient Deep Feature Matching](https://github.com/chicleee/EDM) | Efficient matching related to the EfficientLoFTR family | Worth testing after EfficientLoFTR; released deployment support |
| [MatchAnything](https://github.com/zju3dv/MatchAnything) | Broad matching training; ELoFTR and RoMa variants | First round ELoFTR; RoMa variant later |
| [DKM](https://github.com/Parskatt/DKM) | Dense kernelized matching | Established dense baseline; predecessor of RoMa |
| [RoMa](https://github.com/Parskatt/RoMa) | Robust dense matching | Already adapted; heavier experiment |
| [TinyRoMa](https://github.com/Parskatt/RoMa) | Lightweight dense model using XFeat | Practical earlier dense candidate |
| [RoMa v2](https://github.com/Parskatt/romav2) | Newer dense model with overlap/precision predictions | High-priority GPU addition |
| [UFM](https://github.com/UniFlowMatch/UFM) | Unified optical flow and wide-baseline matching | Strong research candidate; large released models, GPU-oriented |
| [ASpanFormer](https://github.com/apple/ml-aspanformer) | Adaptive attention spans | Secondary detector-free baseline |
| [MatchFormer](https://github.com/InSAI-Lab/MatchFormer) | Hierarchical matching transformer | Secondary baseline |
| [TopicFM / TopicFM+](https://github.com/TruongKhang/TopicFM) | Topic-assisted matching | Secondary family |
| [SE2-LoFTR](https://github.com/georg-bn/se2-loftr) | Rotation-aware LoFTR | Lower priority for translation-only acquisitions |
| [CasMTR](https://github.com/ewrfcas/CasMTR) | Cascaded matching transformers | Secondary candidate |
| [PATS](https://github.com/zju3dv/pats) | Patch-area transport and subdivision | Secondary candidate for scale/patch correspondence |
| [ASTR](https://github.com/ASTR2023/ASTR) | Adaptive spot-guided transformer | Secondary candidate |
| [GLU-Net / GOCor / PDC-Net / PDC-Net+](https://github.com/PruneTruong/DenseMatching) | Global/local dense correspondence and uncertainty | Useful dense research baselines |
| [MASt3R](https://github.com/naver/mast3r) | 3D-aware dense descriptors/matching | Expensive research comparison; not my first choice for planar translation |
| [MINIMA](https://github.com/LSXI7/MINIMA) | Modality-invariant training across matching backbones | Use LoFTR/RoMa/TinyRoMa variants to avoid SuperPoint |
| [GIM](https://github.com/xuelunshen/gim) | Generalized matching learned from Internet videos | Test DKM/LoFTR variants; inspect extractor in LightGlue variants |
| [XoFTR](https://github.com/OnderT/XoFTR) | Cross-modal visible/thermal matching | Specialist; lower priority for same-modality tiles |
| [ReDFeat](https://arxiv.org/abs/2205.07439) | Multimodal learned detection/description | Specialist cross-modal alternative |
| [EDM for equirectangular images](https://github.com/jdk9405/EDM) | Omnidirectional dense matching | Specialized projection; different project from Efficient Deep Feature Matching |

UFM has several released large model variants. Its advertised forthcoming Tiny model should not be treated as an available lightweight option. Treat dense-model throughput figures as hardware- and resolution-specific rather than expected application performance. [UFM implementation and release status](https://github.com/UniFlowMatch/UFM)

## Learned components, refiners, and adjacent approaches

These broaden the inventory, but should not all appear as interchangeable complete stitching engines.

| Component | Role and qualification |
|---|---|
| [LightGlue](https://github.com/cvg/LightGlue) | Learned assignment for compatible feature families; choose ALIKED/DISK rather than SuperPoint |
| [LighterGlue](https://github.com/verlab/accelerated_features) | Lightweight learned matcher trained for XFeat |
| [HardNet / HardNet++](https://github.com/DagnyT/hardnet) | Learned patch descriptors; require a detector |
| [SOSNet](https://github.com/scape-research/SOSNet) | Learned patch descriptors; require a detector |
| [ContextDesc](https://github.com/lzx551402/contextdesc) | Context-enhanced learned descriptors; legacy setup |
| [Key.Net](https://github.com/axelBarroso/Key.Net) | Detector combining handcrafted and learned filters; pair with learned descriptors; hybrid rather than fully learned |
| [MATCHA](https://github.com/feixue94/matcha) | Unified geometric/semantic/temporal descriptors; can use a non-SuperPoint detector such as DISK; distinct from MatchAnything |
| [Patch2Pix](https://github.com/GrumpyZhou/patch2pix) | Learned patch-to-pixel correspondence refinement |
| [Keypt2Subpx](https://github.com/KimSinjeong/keypt2subpx) | Learned subpixel refinement; choose its ALIKED/XFeat/DeDoDe-compatible route |
| [Rotation steerers](https://github.com/georg-bn/rotation-steerers) | Learned descriptor transformation for rotation robustness; not a complete detector |
| [Affine steerers](https://github.com/georg-bn/affine-steerers) | Learned descriptor transformations for affine changes |
| [RAFT](https://github.com/princeton-vl/RAFT) | Learned optical flow; useful adjacent research for overlapping scan imagery, requires overlap/occlusion filtering |
| [GMFlow](https://github.com/haofeixu/gmflow) | Optical flow through global matching; same adaptation caveat |
| [UniMatch](https://github.com/autonomousvision/unimatch) | Unified flow/stereo/depth architecture; use flow correspondences rather than assume a drop-in stitcher |

Do not present **OmniGlue** as a SuperPoint replacement: its released pipeline explicitly uses SuperPoint alongside DINOv2. MINIMA's LightGlue configuration also uses SuperPoint. **SuperGlue** in this app retains SuperPoint. **GlueStick** is a learned joint point/line matcher, so audit its configured feature extractor before considering it SuperPoint-free. [OmniGlue](https://github.com/google-research/omniglue), [MINIMA](https://github.com/LSXI7/MINIMA), [GlueStick](https://github.com/cvg/GlueStick).

## Licensing findings relevant to selecting implementations

These are source observations, not a complete legal clearance of every dependency, training dataset, and weight file. Pin the actual source commit and checkpoint before distribution.

- LightGlue explicitly documents Apache-2.0 for its code and pretrained matcher weights, Apache-2.0 for DISK, and BSD-3-Clause for ALIKED. SuperPoint has separate terms. [Upstream license notes](https://github.com/cvg/LightGlue)
- Current EfficientLoFTR and MatchAnything default branches use **Project Registration License v1.0**. The text permits personal/research/evaluation use without registration and requires registration for organizational project use. These current branches must not simply be labeled Apache-2.0. Earlier vendored commits may carry different terms; inspect those exact files before changing local labels. [EfficientLoFTR license](https://github.com/zju3dv/EfficientLoFTR/blob/main/LICENSE), [MatchAnything license](https://github.com/zju3dv/MatchAnything/blob/main/LICENSE)
- LoMa documents MIT licensing with an Apache-2.0 matcher component. RDD's repository is Apache-2.0. RoMa v2's repository is MIT; backbone and checkpoint terms still deserve separate review. [LoMa](https://github.com/davnords/LoMa), [RDD](https://github.com/xtcpete/rdd), [RoMa v2](https://github.com/Parskatt/romav2)
- MASt3R uses noncommercial CC BY-NC-SA terms and additional weight/dataset conditions. R2D2 also has noncommercial terms. SiLK's GPL-3.0 is a different kind of restriction, not a noncommercial license. [MASt3R](https://github.com/naver/mast3r), [R2D2](https://github.com/naver/r2d2), [SiLK](https://github.com/facebookresearch/silk)
- RIPE includes multiple license files; UPAL includes separately licensed dependencies. LiftFeat's licensing was not established by this review. Mark these unresolved instead of assigning a permissive label by assumption. [RIPE](https://github.com/fraunhoferhhi/RIPE), [UPAL](https://github.com/francois141/upal), [LiftFeat](https://github.com/lyp-deeplearning/LiftFeat)

## How to find which model actually fixes the grid

A correctly arranged filename grid verifies intended positions, not shared image content. A different model may recover missed matches, but cannot create genuine overlap. Repeated texture can also produce convincing but incorrect matches.

For the reported 77-tile run, prioritize the problematic vertical boundaries between rows 03/04 and 05/06: 11 pairs on each boundary, 22 pairs total. These are the links separating the observed 44/22/11-image groups. Include known successful horizontal/vertical neighbors and deliberately nonoverlapping negatives.

1. Save this exact pair list as a reproducible evaluation set. Inspect whether each pair really overlaps; record manual landmarks where possible.
2. Compare each model on the same image resolution, image orientation, and geometric model. Run translation-only as the primary scan experiment; keep a separate homography experiment if useful.
3. Record exact model/checkpoint, preprocessing, feature/match limits, thresholds, warm and cold runtime, and memory. Learned confidence scores are not calibrated identically across methods.
4. Measure correct recovered boundary links, false accepted links, translation residuals, spatial distribution of inliers, and full-grid consistency. More raw matches is not evidence of better registration.
5. Preserve candidate matches, geometric inlier masks, rejection reasons, and estimated shifts for the existing side-by-side debugger. The current confidence-colored match preview is not a geometric inlier visualization; adding explicit inlier/outlier coloring would improve model comparisons.
6. Run the strongest candidates over all 136 intended neighbor pairs. Require a connected graph supported by plausible shifts and consistent grid cycles; a single false bridge can connect an otherwise wrong mosaic.

Potential adapter confounders to fix before a fair benchmark:

- Current alternative-engine preprocessing commonly converts to grayscale, then repeats channels for RGB models. Test native RGB where the pretrained model expects it; keep a documented grayscale experiment for grayscale sources.
- Preserve original pixel coordinates through each model's resize, padding, and normalized-coordinate conventions.
- Cache per-image features for sparse methods. Dense pairwise models have different costs and cannot be compared using feature-extraction timing alone.
- Dense correspondences need spatial sampling and confidence filtering. Existing minimum match/inlier thresholds were not calibrated uniformly for every model family.
- EfficientLoFTR's prepared-image data differs from classic feature records; exposing it in a dropdown alone does not ensure diagnostics work.

## Recommended implementation order after research

1. Expose the already-adapted learned alternatives in the classic grid workflow, retaining translation-only alignment and pair diagnostics. Validate readiness and weight loading per model.
2. Unify EfficientLoFTR/MatchAnything diagnostic adapters and add a saved-pair comparison mode, so a rejected grid edge can be retried across models without rebuilding the full mosaic.
3. Add LoMa B/B128, revised RDD + its compatible LightGlue, and DeDoDe v2.
4. Add RoMa v2 and UFM in a GPU-capable environment. Compare them against TinyRoMa/RoMa before accepting their extra cost.
5. Keep older and specialist approaches in an experimental catalog instead of making the main selector unmanageable.

[vismatch](https://github.com/gmberton/vismatch), formerly image-matching-models, offers a shared interface to many published matching configurations and is useful for exploratory evaluation. Its advertised model count includes variants and pairings, not that many independent algorithm families. Use an isolated environment and preserve upstream licenses/checkpoints; a wrapper does not remove those constraints.

No application behavior or model installation was changed as part of this research report.
