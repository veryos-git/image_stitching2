import numpy as np
from scipy.optimize import least_squares


def components(n,edges):
    adj=[[] for _ in range(n)]
    for e in edges:
        if e['status']=='accepted':adj[e['i']].append(e['j']);adj[e['j']].append(e['i'])
    groups=[];seen=set()
    for i in range(n):
        if i in seen:continue
        group=[i];seen.add(i)
        for j in group:
            for k in adj[j]:
                if k not in seen:group.append(k);seen.add(k)
        groups.append(sorted(group))
    return groups


def solve(n,edges,outlier_px=8.):
    positions=np.zeros((n,2));residuals=[]
    groups=components(n,edges)
    for group in groups:
        if len(group)==1:continue
        indices={i:k for k,i in enumerate(group)}
        ee=[e for e in edges if e['status']=='accepted' and e['i'] in indices]
        matrix=np.zeros((len(ee),len(group)-1));measure=[];weights=[]
        for k,e in enumerate(ee):
            for tile,sign in [(e['i'],-1),(e['j'],1)]:
                if indices[tile]:matrix[k,indices[tile]-1]=sign
            measure.append(e['displacement']);weights.append(1/max(e.get('uncertainty_px') or 1.,.5))
        measure=np.array(measure);weights=np.clip(weights,.1,2.)
        initial=np.linalg.lstsq(matrix,measure,rcond=None)[0]
        def residual(x):return ((matrix@x.reshape(-1,2)-measure)*weights[:,None]).ravel()
        fit=least_squares(residual,initial.ravel(),loss='huber',f_scale=2.,max_nfev=100)
        pp=np.vstack([np.zeros(2),fit.x.reshape(-1,2)])
        for i,k in indices.items():positions[i]=pp[k]
        for k,e in enumerate(ee):
            r=float(np.linalg.norm(matrix[k]@fit.x.reshape(-1,2)-measure[k]))
            residuals.append(dict(i=e['i'],j=e['j'],residual_px=r,flagged=r>outlier_px))
    return dict(positions=positions.tolist(),components=groups,residuals=residuals,qc_consistent=not any(r['flagged'] for r in residuals),uncertainty='unavailable; robust-fit residuals are not calibrated position uncertainty')
