"""Two-stage stitching of ordered tiles into rows, then rows into a mosaic."""
import time
from dataclasses import replace

import cv2
import numpy as np

from .progress import ProgressReporter
from .utils import encode_b64


def stitch_rows(parent, images, grays=None):
    from .stitcher import PanoramaStitcher, StitchError, StitchResult

    cfg = parent.config
    started = time.time()
    positions = cfg.grid_positions
    if len(images) < 2:
        raise StitchError("Need at least 2 images to stitch a panorama.")
    if not positions or len(positions) != len(images):
        raise StitchError("Rows-first stitching requires a row and column for every image.")
    if len(set(map(tuple, positions))) != len(positions):
        raise StitchError("Rows-first stitching requires unique row and column positions.")
    if grays is not None and len(grays) != len(images):
        raise StitchError("images/gray length mismatch")
    groups = {}
    for i, (row, col) in enumerate(positions):
        groups.setdefault(row, []).append((col, i))
    rows = sorted(groups)
    stages = [dict(marker=f"row_{r}", label=f"Step 1 · Stitch row {r}",
                   weight=0.65 / len(rows)) for r in rows]
    stages.append(dict(marker="combine_rows", label="Step 2 · Combine rows", weight=0.35))
    rep = ProgressReporter(emit=parent.reporter._send, stages=stages)
    rep.plan(dict(engine=cfg.engine, alignment=cfg.alignment,
                  stitching_mode="rows_first", num_images=len(images),
                  input_names=cfg.input_names))

    def run(stage, batch, names, masks=None, intermediate=False):
        def forward(event):
            if event['type'] == 'step':
                rep.stage_progress(stage, event['progress'], event['message'])
            elif event['type'] == 'log':
                rep.log(event['message'], event['level'], stage)
            elif event['type'] == 'image':
                rep.image(stage, event['image'], event['label'], event['caption'])

        child_cfg = replace(cfg, stitching_mode="grid",
                            input_max_width=0, preprocessing="none",
                            pairing="sequential" if intermediate else "unordered",
                            disconnected="largest",
                            grid_positions=None, input_names=names,
                            diagnostics_dir=None, diagnostics_url="",
                            crop=False if intermediate else cfg.crop,
                            # Preserve the same pixel scale across all row images.
                            max_output_dim=float('inf') if intermediate else cfg.max_output_dim)
        child = PanoramaStitcher(child_cfg, ProgressReporter(emit=forward))
        child.engine = parent.engine
        try:
            return child.stitch(batch, input_masks=masks, feather=True)
        finally:
            parent.engine = child.engine

    row_images, row_masks, row_stats = [], [], []
    failed_rows, warnings = [], []
    for row in rows:
        marker = f"row_{row}"
        indices = [i for _, i in sorted(groups[row])]
        rep.stage_start(marker, f"Stitching {len(indices)} tiles in row {row}…")
        if len(indices) == 1:
            panorama = images[indices[0]].copy()
            mask = np.full(panorama.shape[:2], 255, np.uint8)
            stats = dict(num_images=1)
        else:
            names = [cfg.input_names[i] if cfg.input_names else str(i + 1) for i in indices]
            try:
                result = run(marker, [images[i] for i in indices], names, intermediate=True)
            except Exception as exc:
                failed_rows.append(dict(row=row, input_indices=indices, reason=str(exc)))
                warnings.append(f"Row {row} failed: {exc}")
                rep.stage_error(marker, str(exc))
                rep.log(warnings[-1], "warning", marker)
                continue
            # Trim only empty outer padding; keep all source content and its mask.
            x, y, w, h = cv2.boundingRect(result.coverage)
            panorama = result.panorama[y:y+h, x:x+w].copy()
            mask = result.coverage[y:y+h, x:x+w].copy()
            stats = result.stats
        row_images.append(panorama)
        row_masks.append(mask)
        row_stats.append(dict(row=row, input_indices=indices, stats=stats))
        rep.image(marker, encode_b64(panorama, ".jpg", 95), f"Completed row {row}",
                  f"{panorama.shape[1]}×{panorama.shape[0]} px",
                  dict(completed_row=row, width=panorama.shape[1], height=panorama.shape[0]))
        rep.stage_done(marker, f"Row {row} complete")

    marker = "combine_rows"
    if not row_images:
        raise StitchError("No rows could be stitched. " + " ".join(warnings))
    completed_rows = [entry['row'] for entry in row_stats]
    rep.stage_start(marker, f"Stitching {len(row_images)} completed row images…")

    def single_row(index):
        # Honor final size/crop settings for a lone row or fallback result.
        from . import blender
        panorama, mask = row_images[index], row_masks[index]
        if cfg.crop:
            bbox = blender.find_crop_bbox(mask.astype(np.float32) / 255, margin=cfg.crop_margin)
            panorama, mask = blender.crop(panorama, bbox), blender.crop(mask, bbox)
        scale = min(1.0, cfg.max_output_dim / max(panorama.shape[:2]))
        if scale < 1:
            size = (max(1, round(panorama.shape[1]*scale)), max(1, round(panorama.shape[0]*scale)))
            panorama = cv2.resize(panorama, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return StitchResult(panorama, dict(engine=cfg.engine, alignment=cfg.alignment), mask)

    if len(row_images) == 1:
        included = [0]
        result = single_row(0)
    else:
        try:
            result = run(marker, row_images, [f"Row {row}" for row in completed_rows], row_masks)
            included = [i - 1 for i in result.stats['graph']['included']]
            result.stats['row_graph'] = result.stats.pop('graph')
        except Exception as exc:
            # Never invent offsets for disconnected rows. Return the completed
            # row covering the most source pixels; all rows remain downloadable.
            included = [max(range(len(row_images)), key=lambda i: np.count_nonzero(row_masks[i]))]
            result = single_row(included[0])
            warnings.append(f"Could not combine rows: {exc}")
            rep.log(warnings[-1], "warning", marker)

    included_rows = [completed_rows[i] for i in included]
    excluded_rows = [row for row in rows if row not in included_rows]
    if excluded_rows:
        warnings.append("Rows omitted from the panorama: " + ", ".join(map(str, excluded_rows)) +
                        ". All completed rows are available separately.")
    included_count = sum(len(row_stats[i]['input_indices']) for i in included)
    result.stats.update(stitching_mode="rows_first", num_images=included_count,
                        input_max_width=cfg.input_max_width, preprocessing=cfg.preprocessing,
                        input_count=len(images), num_rows=len(included_rows), rows=row_stats,
                        completed_rows=completed_rows, included_rows=included_rows,
                        excluded_rows=excluded_rows, failed_rows=failed_rows,
                        partial=bool(excluded_rows), warnings=warnings,
                        output=dict(w=result.panorama.shape[1], h=result.panorama.shape[0]),
                        elapsed=round(time.time() - started, 3))
    rep.stage_done(marker, f"{'Partial result' if excluded_rows else 'Complete'}: "
                   f"{len(included_rows)}/{len(rows)} rows, {included_count}/{len(images)} tiles")
    rep.result(encode_b64(result.panorama, ".jpg", 95), result.panorama.shape[1],
               result.panorama.shape[0], result.stats)
    return result
