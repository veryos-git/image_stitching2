import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import cv2
import numpy as np
from .contracts import RefineResult
from .registration import overlap
from .io import digest

PIXEL_COMMIT='5893fde63ab41a97a20398b6745202ca97aa2b2c'
PIXEL_SHA='9710bcaccac9ae83e888bb194038b7f874a77fde7c939cdfd8ddaefd148ef1f7'


def corner_offsets(H,width,height):
    # Upstream uses boundary corners, TL TR BL BR, not last pixel centers.
    points=np.array([[0,0],[width,0],[0,height],[width,height]],float)
    mapped=np.c_[points,np.ones(4)]@np.asarray(H).T
    return mapped[:,:2]/mapped[:,2:]-points


def validate_result(result):
    if result.status not in ('success','no_improvement','unsupported_input','failed'):raise ValueError('Invalid refinement status')
    if result.frame!='original_tile_pixel_centers_xy':raise ValueError('Missing or unsupported correspondence frame')
    if result.status=='success' and result.points_a is not None:
        a,b=np.asarray(result.points_a),np.asarray(result.points_b)
        if a.ndim!=2 or a.shape[1]!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid refinement correspondence arrays')
    return result


class Identity:
    capabilities={'version':'1','gpu':False,'output':'zero displacement','channels':'registration gray; original channels preserved'}
    def refine(self,req):return RefineResult('success',diagnostics={'identity':True,'deformation':0})


class Farneback:
    capabilities={'version':'opencv-'+cv2.__version__,'gpu':False,'output':'correspondences'}
    def refine(self,req):
        crops=overlap(req.a,req.b,req.displacement)
        if crops is None:return RefineResult('unsupported_input',diagnostics={'reason':'no_overlap'})
        a,b,origin=crops
        if min(a.shape)<24:return RefineResult('unsupported_input',diagnostics={'reason':'overlap_too_narrow'})
        f=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,25,5,7,1.5,0)
        backward=cv2.calcOpticalFlowFarneback(b,a,None,.5,3,25,5,7,1.5,0)
        yy,xx=np.mgrid[8:a.shape[0]-8:12,8:a.shape[1]-8:12].astype(np.float32)
        flow=f[yy.astype(int),xx.astype(int)];targets=np.stack([xx,yy],-1)+flow
        reverse=cv2.remap(backward,targets[...,0],targets[...,1],cv2.INTER_LINEAR)
        texture=cv2.GaussianBlur(a.astype(np.float32)**2,(9,9),0)-cv2.GaussianBlur(a.astype(np.float32),(9,9),0)**2
        good=(np.linalg.norm(flow+reverse,axis=-1)<1)&(texture[yy.astype(int),xx.astype(int)]>16)&(targets[...,0]>2)&(targets[...,0]<a.shape[1]-3)&(targets[...,1]>2)&(targets[...,1]<a.shape[0]-3)
        pa=np.stack([xx,yy],-1)[good]+origin;pb=targets[good]+origin-req.displacement
        if len(pa)<12:return RefineResult('no_improvement',diagnostics={'reason':'insufficient_forward_backward_consistent_points'})
        return RefineResult('success',pa,pb,{'forward_backward_threshold_px':1.,'points':len(pa),'crop_origin':list(origin),'sampling_step':12})


class PixelStitch:
    capabilities={'version':'1','upstream_commit':PIXEL_COMMIT,'checkpoint_sha256':PIXEL_SHA,'gpu':False,'output':'shared-frame bidirectional correspondences','microscopy_validated':False}
    def refine(self,req):
        root=Path(__file__).resolve().parents[1]
        repo=root/'vendor/PixelStitch';checkpoint=repo/'checkpoints/ckpt.pth'
        if not checkpoint.exists():return RefineResult('failed',diagnostics={'reason':'PixelStitch not installed','setup':f'git clone https://github.com/MakiseChris666/PixelStitch {repo}; git -C {repo} checkout {PIXEL_COMMIT}'})
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if commit!=PIXEL_COMMIT or digest(checkpoint)!=PIXEL_SHA:return RefineResult('failed',diagnostics={'reason':'PixelStitch source/checkpoint pin mismatch'})
        with tempfile.TemporaryDirectory(prefix='microscope-pixel-') as temp:
            src=Path(temp)/'request.npz';dest=Path(temp)/'result.npz'
            np.savez(src,a=req.a,b=req.b,d=req.displacement)
            try:
                proc=subprocess.run([sys.executable,str(root/'microscopy/pixel_worker.py'),str(repo),str(src),str(dest)],capture_output=True,text=True,timeout=req.config.get('timeout_seconds',90))
                if proc.returncode:return RefineResult('failed',diagnostics={'reason':'worker_error','stderr':proc.stderr[-3000:]})
                with np.load(dest,allow_pickle=False) as data:
                    return RefineResult('success',data['a'],data['b'],{'upstream_commit':commit,'checkpoint_sha256':PIXEL_SHA,'mapping':'q -> upstream normalized flow grid -> align_corners=False pixel centers -> inverse coarse translation','preprocessing':json.loads(str(data['metadata']))})
            except subprocess.TimeoutExpired:
                return RefineResult('failed',diagnostics={'reason':'timeout','timeout_seconds':req.config.get('timeout_seconds',90)})
            except Exception as exc:return RefineResult('failed',diagnostics={'reason':str(exc)})

REFINERS={'identity':Identity,'farneback':Farneback,'pixelstitch':PixelStitch}
