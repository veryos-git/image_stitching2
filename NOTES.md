# Project notes / backlog

## Blending improvements (user request)

User feedback:

- The AI feature matching is good, but the **blending** could still be better.
- User tested many stitching executables; most of them "failed miserably".
- The one that stood out as great was **AutoPano Giga** (Kolor, later GoPro).

### Reference quality: AutoPano Giga

- Closed source → **no reverse engineering** (and not needed).
- The techniques that made it feel good are all published in open literature,
  so they can be re-implemented independently:

  1. **Graph-cut optimal seam finding** — instead of blending the whole overlap
     (which causes ghosting / double-exposure under parallax), find an optimal
     seam through the overlap (cost = gradient magnitude + color difference) and
     blend only along a narrow band around it. This is the #1 candidate for
     improving our output.
     Refs: Kwatra et al. "Graphcut Textures" (2003); Agarwala et al.
     "Interactive Digital Photomontage" (2004); Brown & Lowe 2007.

  2. **Multi-band (Laplacian) blending along the seam** — we already have
     multi-band blending, but currently apply it to the entire overlap; pairing
     it with a graph-cut seam is the standard "seamless" combo.

  3. **Color / exposure correction** — we have per-image gain; could be upgraded
     to per-channel gains + local vignetting/falloff correction.

  4. **Projection selection** — cylindrical / spherical / stereographic to reduce
     distortion for very wide panoramas (we currently do planar only).

  5. **Global bundle adjustment** — we do sequential chaining (drift can
     accumulate); a global refinement over all pairwise homographies would help
     long / looped sequences.

### Concrete next steps (prioritized)

- [ ] Add a graph-cut seam finder to `stitcher/blender.py`, then restrict the
      multi-band blend to a narrow band around the seam.
- [ ] Per-channel exposure gains.
- [ ] Optional cylindrical / spherical projection.
- [ ] Global bundle adjustment for long incremental sequences.

### Status

- Not started. The current pipeline already has: RANSAC homographies, optional
  ECC refinement, exposure (single gain/image), multi-band blending, auto-crop.
