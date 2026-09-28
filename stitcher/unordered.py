"""Discover overlap without relying on input order; place a maximum spanning tree."""
from itertools import combinations
import cv2
import numpy as np
from .geometry import estimate_homography, homography_inliers, transform_points


def plausible_transform(H, source, target):
    """Reject horizon crossings, flipped/collapsed footprints and no overlap."""
    h, w = source
    th, tw = target
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    den = np.c_[corners, np.ones(4)] @ H[2]
    if not np.isfinite(H).all() or np.any(np.abs(den) < 1e-8) or not (np.all(den > 0) or np.all(den < 0)):
        return False
    warped = transform_points(corners, H).astype(np.float32)
    if not np.isfinite(warped).all() or not cv2.isContourConvex(warped):
        return False
    area = cv2.contourArea(warped, oriented=True)
    if not .1 * w * h < area < 10 * w * h:
        return False
    target_quad = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    overlap, _ = cv2.intersectConvexConvex(warped, target_quad)
    return overlap / min(area, tw * th) >= .03


def reliable_edge(x, y, shape0, shape1, threshold, alignment="homography", details=None):
    if len(x) < 12:
        return None, 0, 'fewer than 12 matches'
    if alignment == "translation":
        from .translation import reliable_translation
        from .geometry import translation_matrix
        offset, count = reliable_translation(x, y, shape0, shape1, threshold, round_offset=False, details=details)
        if offset is None:
            return None, count, "no reliable translation-only overlap (same scale and orientation required)"
        return translation_matrix(*offset), count, None
    try:
        H, _ = estimate_homography(x, y, threshold)
        if H is None:
            return None, 0, 'no homography'
        count, mask = homography_inliers(x, y, H)
        if details is not None: details['transform'] = H
        if count < 12 or count / len(x) < .25:
            return None, count, 'insufficient inlier support (12 and 25% required)'
        for points, shape in ((x[mask], shape0), (y[mask], shape1)):
            area = cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))
            if area / (shape[0] * shape[1]) < .01:
                return None, count, 'inliers concentrated in a small region'
        if not plausible_transform(H, shape0, shape1) or not plausible_transform(np.linalg.inv(H), shape1, shape0):
            return None, count, 'implausible image footprint or overlap'
        return H, count, None
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return None, 0, 'degenerate geometry'


def align_unordered(engine, features, config, reporter, images=None):
    from .stitcher import StitchError
    n = len(features)
    pairs = list(combinations(range(n), 2))
    if config.pairing == "grid":
        positions = config.grid_positions
        if positions is None or len(positions) != n or len({tuple(p) for p in positions}) != n:
            raise StitchError("Grid stitching requires a unique row and column for every image.")
        pairs = [(i, j) for i, j in pairs
                 if abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1]) == 1]
    edges, diagnostics = [], []
    reporter.stage_start('match', f'Searching {len(pairs)} {"grid neighbor" if config.pairing == "grid" else "image"} pairs…')
    for k, (i, j) in enumerate(pairs):
        reporter.stage_progress('match', k / len(pairs), f'Overlap search {k+1}/{len(pairs)}: image {i+1} ↔ {j+1}')
        from .diagnostics import evaluate_pair
        try:
            H, diagnostic, preview = evaluate_pair(engine, features[i], features[j], config, [i+1,j+1])
        except Exception as exc:
            H, preview = None, None
            diagnostic = dict(pair=[i+1,j+1],matches=0,inliers=0,accepted=False,
                engine=config.engine,engine_options=config.engine_options,reason=str(exc),
                rejection=dict(code='matching_failed',message=str(exc)))
            if config.diagnostics_dir:
                import json
                from pathlib import Path
                Path(config.diagnostics_dir).mkdir(parents=True,exist_ok=True)
                (Path(config.diagnostics_dir)/f'{i+1}-{j+1}.json').write_text(json.dumps(diagnostic))
        diagnostics.append(diagnostic)
        names = [config.input_names[t] if config.input_names else f'Image {t+1}' for t in (i,j)]
        reporter._send(dict(type='pair_diagnostic', diagnostic=diagnostic, names=names))
        if config.viz:
            if preview is None:
                from .utils import draw_matches, encode_b64
                preview = encode_b64(draw_matches(features[i].small_gray,features[j].small_gray,
                    np.empty((0,2)),np.empty((0,2))))
            reporter.image('match', preview, f'Matches {i+1} ↔ {j+1}', diagnostic['reason'] or 'Accepted overlap',
                {'pair_diagnostic':diagnostic,'drawn_matches':min(diagnostic['matches'],500),'names':names})
        if H is not None:
            edges.append((diagnostic['inliers'],i,j,H))
    retries, priors = [], []
    if config.guided_retry or config.grid_priors:
        from .enhancements import recover_edges
        edges, retries, priors = recover_edges(engine, features, images, config, reporter, pairs, edges, diagnostics)
    reporter.stage_done('match', f'{len(edges)-len(priors)}/{len(pairs)} reliable overlaps', {'matches_per_pair':[p['matches'] for p in diagnostics]})
    reporter.stage_start('align', 'Building overlap graph…')
    # Kruskal: highest-inlier edges form a maximum spanning forest.
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tree = []
    for count, i, j, H in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        a, b = root(i), root(j)
        if a != b:
            parent[a] = b
            tree.append((count, i, j, H))
    groups = {}
    for i in range(n): groups.setdefault(root(i), []).append(i)
    components = sorted(groups.values(), key=lambda c: (-len(c), -sum(e[0] for e in edges if e[1] in c), c[0]))
    included = components[0]
    excluded = [i for i in range(n) if i not in included]
    graph = dict(mode=config.pairing, grid_positions=config.grid_positions, pairs=diagnostics, components=[[i+1 for i in c] for c in components], included=[i+1 for i in included], excluded=[i+1 for i in excluded], tested_pairs=len(pairs), accepted_pairs=len(edges))
    graph.update(accepted_pairs=len(edges)-len(priors), guided_retries=retries, predicted_edges=priors,
                 placement_includes_predictions=bool(priors), global_adjustment=config.global_adjustment)
    if config.diagnostics_dir:
        import json
        from pathlib import Path
        (Path(config.diagnostics_dir)/'graph.json').write_text(json.dumps(graph,indent=2))
    reporter.stage_done('align', f'{len(components)} overlap group(s); {len(included)}/{n} images in largest group', {'inliers_per_pair':[p['inliers'] for p in diagnostics], 'graph':graph})
    if len(included) < 2 or (excluded and config.disconnected == 'reject'):
        message = 'No reliable overlap found. These images cannot be stitched into one panorama.' if len(included) < 2 else f'Images form {len(components)} disconnected groups. Use “Largest connected group” to produce a partial panorama.'
        if config.pairing == 'grid':
            message = f'Grid images form {len(components)} disconnected groups. Check the detected row and column positions and upload overlapping horizontal or vertical neighbors.'
            if config.alignment == 'translation':
                message += ' Translation-only alignment requires the same scale and orientation.'
        error = StitchError(message)
        error.graph = graph
        raise error
    adjacency = {i:[] for i in included}
    for count, i, j, H in tree:
        if i in adjacency:
            adjacency[i].append((j, np.linalg.inv(H)))
            adjacency[j].append((i, H))
    # Anchor at the tree center to limit accumulated transform chains.
    def eccentricity(start):
        distances = {start:0}; queue = [start]
        for i in queue:
            for j, _ in adjacency[i]:
                if j not in distances: distances[j]=distances[i]+1; queue.append(j)
        return max(distances.values())
    reference = min(included, key=lambda i: (eccentricity(i), i))
    if config.pairing == "grid" and config.reference == "first":
        reference = min(included)
    transforms = {reference:np.eye(3)}; queue = [reference]
    for i in queue:
        for j, to_i in adjacency[i]:
            if j not in transforms:
                transforms[j] = transforms[i] @ to_i
                transforms[j] /= transforms[j][2,2]
                queue.append(j)
    graph['reference'] = reference + 1
    graph['tree_pairs'] = [[i+1,j+1] for _,i,j,_ in tree if i in adjacency]
    if config.global_adjustment:
        from .enhancements import solve_translations
        from .geometry import translation_matrix
        positions, _ = solve_translations(n, edges)
        transforms = {i: translation_matrix(*(positions[i]-positions[reference])) for i in included}
    if config.global_adjustment or config.grid_priors or config.guided_retry:
        graph['positions'] = {str(i+1): transforms[i][:2,2].tolist() for i in included}
        graph['layout_residuals'] = [dict(pair=[i+1,j+1], residual_px=float(np.linalg.norm(
            transforms[j][:2,2]-transforms[i][:2,2]+H[:2,2])))
            for _, i, j, H in edges if i in transforms and j in transforms]
        if graph['layout_residuals'] and max(d['residual_px'] for d in graph['layout_residuals']) > 25:
            reporter.log('Layout residual exceeds 25 pixels; inspect the graph diagnostics.', 'warning', 'align')
    if config.diagnostics_dir:
        (Path(config.diagnostics_dir)/'graph.json').write_text(json.dumps(graph,indent=2))
    return [transforms[i] for i in included], included, reference, graph
