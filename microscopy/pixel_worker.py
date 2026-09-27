"""Isolated PixelStitch inference: avoids its generic `modules` import collision."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    repo,src,dest=map(Path,sys.argv[1:]);sys.path.insert(0,str(repo))
    from modules.models.raft.raft import RAFTStitch
    from modules.models.utils import getAdaptiveWeight, getGrid
    torch.set_num_threads(4)
    class Args(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    model=RAFTStitch(Args(ftdown=8,radius=6))
    model.load_state_dict(torch.load(repo/'checkpoints/ckpt.pth',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    data=np.load(src);a,b,d=data['a'],data['b'],data['d']
    if a.shape!=b.shape:raise ValueError('PixelStitch adapter requires equal tile dimensions')
    # Standard center-coordinate translation initialization into one shared frame.
    lo=np.floor(np.minimum([0,0],d));hi=np.ceil(np.maximum([a.shape[1],a.shape[0]],d+[b.shape[1],b.shape[0]]))
    width,height=(hi-lo).astype(int);scale=min(1.,512/max(width,height))
    w,h=max(16,round(width*scale)),max(16,round(height*scale));sx,sy=w/width,h/height
    yy,xx=np.mgrid[:h,:w].astype(np.float32);q=np.stack([(xx+.5)/sx-.5+lo[0],(yy+.5)/sy-.5+lo[1]],-1)
    images=[]
    for im,position in ((a,np.zeros(2)),(b,d)):
        warped=cv2.remap(im,(q[...,0]-position[0]).astype(np.float32),(q[...,1]-position[1]).astype(np.float32),cv2.INTER_LINEAR)
        images.append(torch.from_numpy(warped).float()[None].expand(3,-1,-1)/255)
    pair=torch.stack(images);dh=(-h)%8;dw=(-w)%8
    pair=F.pad(pair,(dw//2,dw-dw//2,dh//2,dh-dh//2))
    with torch.inference_mode():flows,_=model(pair,iters=4)
    flow=flows[-1][:,:,dh//2:dh//2+h,dw//2:dw//2+w].permute(0,2,3,1)
    center=np.array([a.shape[1]/2,a.shape[0]/2]);c1=torch.tensor(((center-lo)*[sx,sy])[None],dtype=torch.float32);c2=torch.tensor(((center+d-lo)*[sx,sy])[None],dtype=torch.float32)
    weights=getAdaptiveWeight(flow[[0]],flow[[1]],c1,c2)
    grid=getGrid(h,w);points=[]
    for index,weight in enumerate(weights):
        normalized=grid+flow[[index]]*weight/torch.tensor([w/2,h/2])
        # Exact grid_sample default align_corners=False inverse normalization.
        sampled=((normalized[0].numpy()+1)*[w/2,h/2]-.5)
        original=(sampled+.5)/[sx,sy]-.5+lo-(np.zeros(2) if index==0 else d)
        points.append(original[8:-8:12,8:-8:12].reshape(-1,2))
    pa,pb=points;good=np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
    for p,im in ((pa,a),(pb,b)):good&=(p[:,0]>3)&(p[:,1]>3)&(p[:,0]<im.shape[1]-4)&(p[:,1]<im.shape[0]-4)
    if good.sum()<12:raise ValueError('Insufficient shared-frame support')
    metadata=dict(canvas_origin=lo.tolist(),scale_xy=[sx,sy],padding=[dw//2,dh//2],input='gray replicated to three channels, [0,1]',iterations=4,coarse_a_to_b=[[1,0,float(-d[0])],[0,1,float(-d[1])],[0,0,1]],corner_displacements_TL_TR_BL_BR=np.tile(-d,(4,1)).tolist(),coarse_warp_policy='engine translation into shared center-coordinate canvas; upstream homography resampler replaced explicitly')
    np.savez(dest,a=pa[good],b=pb[good],metadata=json.dumps(metadata))

if __name__=='__main__':main()
