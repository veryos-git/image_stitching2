"""Replaceable translation registrars; scores are diagnostics, not probabilities."""
import cv2
import numpy as np
from .contracts import CoarseResult


def overlap(a,b,d):
    dx,dy=map(float,d)
    x0=max(0,int(np.ceil(dx))); y0=max(0,int(np.ceil(dy)))
    x1=min(a.shape[1],int(np.floor(dx+b.shape[1]))); y1=min(a.shape[0],int(np.floor(dy+b.shape[0])))
    if x1-x0<8 or y1-y0<8: return None
    yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    bb=cv2.remap(b,(xx-dx).astype(np.float32),(yy-dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return a[y0:y1,x0:x1],bb,(x0,y0)


def assess(a,b,d,req=None):
    crops=overlap(a,b,d)
    if crops is None: return {'ncc':-1.,'texture':0.,'overlap_pixels':0}
    aa,bb,origin=crops;aa=aa.astype(float);bb=bb.astype(float)
    if req is not None and (req.mask_a is not None or req.mask_b is not None):
        x,y=origin;valid=np.ones(aa.shape,bool)
        if req.mask_a is not None:valid &= req.mask_a[y:y+aa.shape[0],x:x+aa.shape[1]]
        if req.mask_b is not None:
            yy,xx=np.mgrid[y:y+aa.shape[0],x:x+aa.shape[1]].astype(np.float32)
            valid &= cv2.remap(req.mask_b.astype(np.float32),xx-float(d[0]),yy-float(d[1]),cv2.INTER_LINEAR)>=.999
        aa,bb=aa[valid],bb[valid]
        if aa.size<64:return {'ncc':-1.,'texture':0.,'overlap_pixels':int(aa.size)}
    sa,sb=aa.std(),bb.std()
    ncc=float(np.mean((aa-aa.mean())*(bb-bb.mean()))/max(sa*sb,1e-9))
    return dict(ncc=ncc,texture=float(min(sa,sb)),overlap_pixels=int(aa.size))


def plausible(d,req):
    dx,dy=d;h,w=req.a.shape
    dr,dc=np.sign(req.direction)
    if dc and not (.05*w<dx*dc<.98*w and (dr or abs(dy)<.4*h)): return False
    if dr and not (.05*h<dy*dr<.98*h and (dc or abs(dx)<.4*w)): return False
    if req.prior is not None and np.linalg.norm(np.asarray(d)-req.prior)>req.radius*max(h,w): return False
    return True


class PhaseCorrelation:
    id='phase_correlation'
    capabilities={'geometry':'translation','dtype':'uint8 registration previews','gpu':False,'version':'1'}
    def register(self,req):
        a,b=req.a.astype(np.float32),req.b.astype(np.float32)
        if min(a.std(),b.std())<3: return CoarseResult('failed',reason='textureless_tile')
        # Linear, zero-padded phase correlation avoids circular-wrap placements.
        shape=(a.shape[0]+b.shape[0]-1,a.shape[1]+b.shape[1]-1)
        fa=np.fft.rfft2(a-a.mean(),s=shape);fb=np.fft.rfft2(b-b.mean(),s=shape)
        cross=fa*np.conj(fb);corr=np.fft.irfft2(cross/np.maximum(abs(cross),1e-9),s=shape).real
        yy,xx=np.indices(shape);dx=np.where(xx<=a.shape[1]-1,xx,xx-shape[1]);dy=np.where(yy<=a.shape[0]-1,yy,yy-shape[0])
        dr,dc=np.sign(req.direction)
        valid=np.ones(shape,bool)
        if dc: valid&=(dx*dc>.05*a.shape[1])&(dx*dc<.98*a.shape[1])&((abs(dy)<.4*a.shape[0]) if not dr else True)
        if dr: valid&=(dy*dr>.05*a.shape[0])&(dy*dr<.98*a.shape[0])&((abs(dx)<.4*a.shape[1]) if not dc else True)
        if req.prior is not None:valid&=((dx-req.prior[0])**2+(dy-req.prior[1])**2<(req.radius*max(a.shape))**2)
        corr[~valid]=-np.inf;candidates=[]
        for _ in range(12):
            y,x=np.unravel_index(np.argmax(corr),shape)
            if not np.isfinite(corr[y,x]):break
            d=np.array([dx[y,x],dy[y,x]],float); score=assess(a,b,d,req)
            if score['texture']>=4 and score['overlap_pixels']>=.015*min(a.size,b.size):
                crops=overlap(a,b,d)
                try:
                    shift,response=cv2.phaseCorrelate(crops[0].astype(np.float32),crops[1].astype(np.float32))
                    proposed=d-np.array(shift)
                    if np.linalg.norm(shift)<4 and plausible(proposed,req):
                        improved=assess(a,b,proposed,req)
                        if improved['ncc']>score['ncc']:d=proposed;score=improved
                except cv2.error:pass
                candidates.append(dict(displacement=d.tolist(),scores=score,uncertainty_px=None))
            dist=(dx-d[0])**2+(dy-d[1])**2;corr[dist<64]=-np.inf
        candidates.sort(key=lambda c:-c['scores']['ncc'])
        if not candidates or candidates[0]['scores']['ncc']<.6:return CoarseResult('failed',candidates,'weak_correlation')
        if len(candidates)>1 and candidates[1]['scores']['ncc']>candidates[0]['scores']['ncc']-.04:
            return CoarseResult('ambiguous',candidates,'multiple_plausible_offsets')
        return CoarseResult('accepted',candidates)


class FeatureTranslation:
    id='feature_translation'
    capabilities={'geometry':'translation','dtype':'uint8 registration previews','gpu':False,'version':'1'}
    def register(self,req):
        detector=cv2.SIFT_create(nfeatures=4000)
        ka,da=detector.detectAndCompute(req.a,None);kb,db=detector.detectAndCompute(req.b,None)
        if da is None or db is None:return CoarseResult('failed',reason='no_features')
        raw=cv2.BFMatcher().knnMatch(da,db,k=2)
        matches=[p[0] for p in raw if len(p)==2 and p[0].distance<.7*p[1].distance]
        if len(matches)<8:return CoarseResult('failed',reason='too_few_feature_matches')
        pa=np.array([ka[m.queryIdx].pt for m in matches]);pb=np.array([kb[m.trainIdx].pt for m in matches]);ds=pa-pb
        bins=np.round(ds/3).astype(int);_,counts=np.unique(bins,axis=0,return_counts=True)
        candidates=[]
        for d in ds[np.argsort(np.linalg.norm(ds-np.median(ds,axis=0),axis=1))][:500]:
            if not plausible(d,req):continue
            mask=np.linalg.norm(ds-d,axis=1)<3
            if mask.sum()<8:continue
            d=np.median(ds[mask],axis=0)
            if any(np.linalg.norm(d-np.array(c['displacement']))<6 for c in candidates):continue
            score=assess(req.a,req.b,d,req)
            hull=cv2.contourArea(cv2.convexHull(pa[mask].astype(np.float32)))
            score.update(inliers=int(mask.sum()),matches=len(matches),spread_px2=float(hull))
            if score['texture']<4 or score['ncc']<.45 or hull<.002*req.a.size:continue
            candidates.append(dict(displacement=d.tolist(),scores=score,uncertainty_px=float(np.median(np.linalg.norm(ds[mask]-d,axis=1)))))
        candidates.sort(key=lambda c:-c['scores']['inliers'])
        if not candidates:return CoarseResult('failed',reason='no_supported_translation')
        if len(candidates)>1 and candidates[1]['scores']['inliers']>=.8*candidates[0]['scores']['inliers']:
            return CoarseResult('ambiguous',candidates,'multiple_feature_offsets')
        return CoarseResult('accepted',candidates)

COARSE={'phase_correlation':PhaseCorrelation,'feature_translation':FeatureTranslation}
