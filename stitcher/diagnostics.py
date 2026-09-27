"""Bounded, durable original-pixel pair artifacts shared by grid runs and retries."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .contract import match_pair
from .geometry import transform_points
from .utils import draw_matches, encode_b64

MAX_CORRESPONDENCES = 16384


def spatial_sample(matches, shape, budget):
    """Round-robin across an 8×8 source grid; strongest score first per cell."""
    n = len(matches.points0)
    if n <= budget: return matches
    from .contract import PairMatches
    xy = matches.points0 / np.array([shape[1],shape[0]])
    cells = np.minimum((xy*8).astype(int),7)
    cell_id = cells[:,1]*8+cells[:,0]
    order = np.arange(n) if matches.scores is None else np.argsort(-matches.scores,kind='stable')
    buckets = [order[cell_id[order] == i].tolist() for i in range(64)]
    indices = []
    for depth in range(max(map(len,buckets))):
        for bucket in buckets:
            if depth < len(bucket): indices.append(bucket[depth])
            if len(indices) == budget: break
        if len(indices) == budget: break
    idx = np.asarray(indices)
    return PairMatches(matches.points0[idx],matches.points1[idx],None if matches.scores is None else matches.scores[idx],
                       {**matches.metadata,'sampling':dict(policy='8x8 spatial round-robin, descending native score',seed=0,original_count=n,budget=budget)})


def evaluate_pair(engine, a, b, config, pair):
    from .unordered import reliable_edge
    from .catalog import CATALOG
    started = time.perf_counter()
    m = spatial_sample(match_pair(engine,a,b),a.shape,min(MAX_CORRESPONDENCES,config.engine_options.get('dense_sample_budget',MAX_CORRESPONDENCES)))
    details = {}
    H,count,reason = reliable_edge(m.points0,m.points1,a.shape,b.shape,config.ransac_thresh,config.alignment,details=details)
    candidate = details.get('transform',H)
    residuals = np.full(len(m.points0),np.nan)
    if candidate is not None:
        try:
            residuals = .5*(np.linalg.norm(transform_points(m.points0,candidate)-m.points1,axis=1)+
                             np.linalg.norm(transform_points(m.points1,np.linalg.inv(candidate))-m.points0,axis=1))
        except np.linalg.LinAlgError:
            candidate = None
    threshold = config.ransac_thresh if config.alignment == 'translation' else 4.
    mask = residuals < threshold
    coverage = []
    for pts,shape in ((m.points0[mask],a.shape),(m.points1[mask],b.shape)):
        coverage.append(float(cv2.contourArea(cv2.convexHull(pts))/(shape[0]*shape[1])) if len(pts)>=3 else 0.)
    data = dict(pair=list(pair), names=[a.image_id,b.image_id], matches=len(m.points0),inliers=int(count),
        accepted=H is not None, reason=reason, rejection=None if reason is None else dict(code='geometry_rejected',message=reason),
        engine=config.engine, checkpoint=CATALOG[config.engine]['checkpoint'],engine_options=config.engine_options,
        candidate_transform=None if candidate is None else candidate.tolist(),
        alignment=config.alignment, transform=None if H is None else H.tolist(),coverage=coverage,
        residual_median_px=float(np.median(residuals[mask])) if mask.any() else None,
        elapsed=time.perf_counter()-started,device=str(getattr(engine,'device','cpu')),
        metadata=m.metadata, confidence_available=m.scores is not None,
        positions=[config.grid_positions[i-1] for i in pair] if config.grid_positions else None)
    base = Path(config.diagnostics_dir) if config.diagnostics_dir else None
    key = '-'.join(map(str,pair))
    if base:
        base.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(base/(key+'.npz'),points0=m.points0,points1=m.points1,
            scores=np.empty(0) if m.scores is None else m.scores,inliers=mask,residuals=residuals)
        data['artifact'] = f'{config.diagnostics_url}/{key}.npz' if config.diagnostics_url else str(base/(key+'.npz'))
    preview = None
    previews = {}
    for mode in ('all','inliers','outliers','confidence') if config.viz else ():
        select = mask if mode == 'inliers' else ~mask if mode == 'outliers' else np.ones(len(mask),bool)
        scores = m.scores[select] if mode == 'confidence' and m.scores is not None else None
        img = draw_matches(a.small_gray,b.small_gray,m.points0[select]*a.scale,m.points1[select]*b.scale,scores,max_draw=500)
        if mode == 'all': preview = encode_b64(img)
        if base:
            cv2.imwrite(str(base/f'{key}-{mode}.jpg'),img)
            previews[mode] = f'{config.diagnostics_url}/{key}-{mode}.jpg' if config.diagnostics_url else str(base/f'{key}-{mode}.jpg')
    data['previews'] = previews
    if base: (base/(key+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    return H,data,preview
