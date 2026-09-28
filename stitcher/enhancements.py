"""Optional batch photometry and translation-graph recovery."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


def correct_flatfield(images):
    """Estimate camera-coordinate shading from up to 48 tiles in small bands."""
    if len(images) < 2 or len({im.shape for im in images}) != 1:
        raise ValueError('Flat-field correction requires at least two equal-sized tiles.')
    h, w = images[0].shape[:2]
    samples = [images[i] for i in np.linspace(0, len(images)-1, min(48, len(images)), dtype=int)]
    field = np.empty_like(images[0], dtype=np.float32)
    for y in range(0, h, 160):
        field[y:y+160] = np.median(np.stack([im[y:y+160] for im in samples]), axis=0)
    field = cv2.GaussianBlur(field, (0, 0), max(8, min(h, w)/12))
    field = np.maximum(field / max(float(field.mean()), 1e-6), .15)
    return [np.clip(np.rint(im.astype(np.float32)/field), 0, 255).astype(np.uint8) for im in images]


def solve_translations(n, edges):
    """Six Huber IRLS steps; each disconnected component is anchored separately.

    Edges are (weight, i, j, H_i_to_j), so position[j]-position[i] = -H[:2,2].
    Component labels must be checked before using relative positions.
    """
    adjacency = [[] for _ in range(n)]
    for _, i, j, _ in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    labels = np.full(n, -1, int)
    anchors = []
    for i in range(n):
        if labels[i] >= 0:
            continue
        anchors.append(i)
        labels[i] = i
        queue = [i]
        for node in queue:
            for other in adjacency[node]:
                if labels[other] < 0:
                    labels[other] = i
                    queue.append(other)
    free = [i for i in range(n) if i not in anchors]
    positions = np.zeros((n, 2))
    if not edges or not free:
        return positions, labels
    A = np.zeros((len(edges), n))
    shifts, weights = [], []
    for k, (weight, i, j, H) in enumerate(edges):
        A[k, i], A[k, j] = -1, 1
        shifts.append(-H[:2, 2])
        weights.append(weight)
    A = A[:, free]
    shifts, weights = np.asarray(shifts), np.asarray(weights, float)
    robust = np.ones(len(edges))
    for _ in range(6):
        root = np.sqrt(weights * robust)
        fit = np.linalg.lstsq(A * root[:, None], shifts * root[:, None], rcond=None)[0]
        residual = np.linalg.norm(A @ fit - shifts, axis=1)
        delta = max(2., 1.5 * float(np.median(residual)))
        robust = np.minimum(1., delta / np.maximum(residual, 1e-9))
    positions[free] = fit
    return positions, labels


def predicted_shift(i, j, edges, grid, positions, labels):
    if labels[i] == labels[j]:
        return positions[i] - positions[j], 'layout'
    direction = np.asarray(grid[j]) - grid[i]
    samples = []
    for _, a, b, H in edges:
        d = np.asarray(grid[b]) - grid[a]
        if np.array_equal(d, direction):
            samples.append(H[:2, 2])
        elif np.array_equal(d, -direction):
            samples.append(-H[:2, 2])
    if len(samples) < 2:
        return None, None
    samples = np.asarray(samples)
    median = np.median(samples, axis=0)
    # Do not extrapolate from disagreeing measured grid steps.
    if np.max(np.linalg.norm(samples-median, axis=1)) > max(5., .1*np.linalg.norm(median)):
        return None, None
    return median, 'grid_median'


def crop_matches(engine, images, i, j, shift, radius):
    """Run the same matcher on predicted overlapping crops; return tile coordinates."""
    from .contract import prepare_image, match_pair, PairMatches
    a, b = images[i], images[j]
    bounds = []
    for image, other, delta in ((a, b, shift), (b, a, -shift)):
        lo = np.maximum(0, np.floor(-delta-radius)).astype(int)
        hi = np.minimum([image.shape[1], image.shape[0]],
                        np.ceil([other.shape[1], other.shape[0]]-delta+radius)).astype(int)
        if np.any(hi-lo < 32):
            raise ValueError('Predicted overlap is too small for a matcher retry.')
        bounds.append((lo, hi))
    prepared = [prepare_image(engine, im[lo[1]:hi[1], lo[0]:hi[0]], str(idx+1))
                for im, (lo, hi), idx in zip((a, b), bounds, (i, j))]
    m = match_pair(engine, *prepared)
    return PairMatches(m.points0+bounds[0][0], m.points1+bounds[1][0], m.scores,
                       {**m.metadata, 'retry': 'predicted_overlap', 'predicted_shift': shift.tolist()}).validated(a.shape[:2], b.shape[:2])


def recover_edges(engine, features, images, config, reporter, pairs, edges, diagnostics):
    """Retry failed measured edges first; optionally add explicitly marked weak priors."""
    from .diagnostics import evaluate_pair
    from .geometry import translation_matrix
    measured = list(edges)
    retries, priors = [], []
    pos, labels = solve_translations(len(features), measured)
    failed = [(i, j) for i, j in pairs if not any(a == i and b == j for _, a, b, _ in measured)]
    if config.guided_retry:
        for i, j in failed:
            shift, source = predicted_shift(i, j, measured, config.grid_positions, pos, labels)
            if shift is None:
                continue
            radius = max(8., 2*config.ransac_thresh)
            attempt = dict(pair=[i+1, j+1], prediction=source, accepted=False)
            try:
                m = crop_matches(engine, images, i, j, shift, radius)
                retry_cfg = replace(config,
                    diagnostics_dir=str(Path(config.diagnostics_dir)/'guided') if config.diagnostics_dir else None,
                    diagnostics_url=config.diagnostics_url+'/guided' if config.diagnostics_url else '')
                H, diagnostic, preview = evaluate_pair(engine, features[i], features[j], retry_cfg,
                    [i+1, j+1], matches=m, predicted_shift=shift, prediction_radius=radius)
                attempt.update(diagnostic)
                reporter._send(dict(type='pair_diagnostic', diagnostic=diagnostic,
                    names=diagnostic['names']))
                if H is not None:
                    measured.append((diagnostic['inliers'], i, j, H))
                    diagnostics[:] = [diagnostic if d['pair'] == [i+1, j+1] else d for d in diagnostics]
            except Exception as exc:
                attempt['reason'] = str(exc)
            retries.append(attempt)
            reporter.log(f"Guided retry {i+1}–{j+1}: {'accepted' if attempt['accepted'] else attempt.get('reason', 'rejected')}",
                         marker='match')
    pos, labels = solve_translations(len(features), measured)
    combined = list(measured)
    if config.grid_priors and measured:
        for i, j in failed:
            if any(a == i and b == j for _, a, b, _ in measured):
                continue
            shift, source = predicted_shift(i, j, measured, config.grid_positions, pos, labels)
            if shift is None or source != 'grid_median':
                continue
            h, w = features[i].shape
            th, tw = features[j].shape
            overlap = np.maximum(0, np.minimum(shift+[w,h], [tw,th])-np.maximum(shift, 0))
            if np.prod(overlap) < .03*min(w*h, tw*th):
                continue
            combined.append((.02, i, j, translation_matrix(*shift)))
            priors.append(dict(pair=[i+1,j+1], kind='prior:grid_median', weight=.02,
                               transform=translation_matrix(*shift).tolist()))
        if priors:
            reporter.log(f'{len(priors)} grid links use predicted placement, not measured matches.', 'warning', 'align')
    return combined, retries, priors
