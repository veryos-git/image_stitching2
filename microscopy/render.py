from pathlib import Path
import os
import cv2
import numpy as np
import tifffile
from .io import load_image, save_json, digest, preview
from .mesh import displacement


def render_component(tiles,positions,meshes,out,chunk_size=512,emit=lambda x:None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cache=out/'sources';cache.mkdir(exist_ok=True)
    sources=[];masks=[];bounds=[]
    for i,(tile,p,m) in enumerate(zip(tiles,positions,meshes)):
        if digest(tile['path'])!=tile['sha256']:raise ValueError('Original changed since registration: '+tile['id'])
        if tile.get('mask') and digest(tile['mask'])!=tile['mask_sha256']:raise ValueError('Mask changed since registration')
        path=cache/(tile['sha256']+'.npy')
        if not path.exists():
            temp=cache/f'{i}.tmp.npy';np.save(temp,load_image(tile['path']));os.replace(temp,path)
        sources.append(np.load(path,mmap_mode='r'))
        masks.append(load_image(tile['mask'])>0 if tile.get('mask') else None)
        yy,xx=np.meshgrid(np.linspace(0,tile['height']-1,m.shape[0]),np.linspace(0,tile['width']-1,m.shape[1]),indexing='ij')
        bounds.append(np.stack([xx,yy],-1)+np.asarray(p)+m)
    allpts=np.concatenate([b.reshape(-1,2) for b in bounds])
    lo=np.floor(allpts.min(0));hi=np.ceil(allpts.max(0));w,h=map(int,hi-lo+1)
    if w*h>2_000_000_000:raise ValueError('Canvas exceeds debug-export budget (2 billion pixels)')
    channels=1 if sources[0].ndim==2 else sources[0].shape[2];dtype=sources[0].dtype
    shape=(h,w) if channels==1 else (h,w,channels)
    image_path=out/'mosaic.partial.tif';coverage_path=out/'coverage.partial.tif'
    image=tifffile.memmap(image_path,shape=shape,dtype=dtype,bigtiff=True,photometric='rgb' if channels==3 else 'minisblack',metadata={'axes':'YX' if channels==1 else 'YXC','canvas_origin_xy':lo.tolist(),'pixel_size_um_x':tiles[0].get('pixel_size_um_x'),'pixel_size_um_y':tiles[0].get('pixel_size_um_y')})
    coverage=tifffile.memmap(coverage_path,shape=(h,w),dtype=np.uint16,bigtiff=True)
    total=((h+chunk_size-1)//chunk_size)*((w+chunk_size-1)//chunk_size);done=0
    for y in range(0,h,chunk_size):
        for x in range(0,w,chunk_size):
            ch,cw=min(chunk_size,h-y),min(chunk_size,w-x)
            yy,xx=np.mgrid[y:y+ch,x:x+cw];q=np.stack([xx,yy],-1).astype(float)+lo
            accum=np.zeros((ch,cw,channels),np.float64);weights=np.zeros((ch,cw));counts=np.zeros((ch,cw),np.uint16)
            for tile,p,m,source,mask,bound in zip(tiles,positions,meshes,sources,masks,bounds):
                if np.any(q[-1,-1]<bound.reshape(-1,2).min(0)) or np.any(q[0,0]>bound.reshape(-1,2).max(0)):continue
                base=q-np.asarray(p);sampling=base.copy()
                for _ in range(12):sampling=base-displacement(m,sampling,tile['width'],tile['height'])
                error=np.linalg.norm(sampling+displacement(m,sampling,tile['width'],tile['height'])-base,axis=-1)
                mx,my=sampling[...,0].astype(np.float32),sampling[...,1].astype(np.float32)
                valid=(mx>=0)&(my>=0)&(mx<=tile['width']-1)&(my<=tile['height']-1)&(error<.01)
                if mask is not None:
                    mask_sample=cv2.remap(mask.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                    valid&=mask_sample>=.999
                weight=np.maximum(0,np.minimum.reduce([mx+1,my+1,tile['width']-mx,tile['height']-my])).astype(float)*valid
                samples=cv2.remap(source.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                if channels==1:samples=samples[...,None]
                accum+=samples*weight[...,None];weights+=weight;counts+=valid.astype(np.uint16)
            values=accum/np.maximum(weights[...,None],1e-12)
            if np.issubdtype(dtype,np.integer):values=np.clip(np.rint(values),0,np.iinfo(dtype).max)
            image[y:y+ch,x:x+cw]=values[...,0].astype(dtype) if channels==1 else values.astype(dtype)
            coverage[y:y+ch,x:x+cw]=counts;done+=1
            emit(f'Rendering chunk {done}/{total}')
    image.flush();coverage.flush()
    # Preview is explicitly non-quantitative; original export retains source precision.
    stride=max(1,int(np.ceil(max(w,h)/1600)));small=np.asarray(image[::stride,::stride])
    if channels==3:
        view=small
        if dtype!=np.uint8:
            low,high=np.percentile(view,[1,99]);view=np.clip((view-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8)
        cv2.imwrite(str(out/'preview.jpg'),cv2.cvtColor(view,cv2.COLOR_RGB2BGR))
    else:cv2.imwrite(str(out/'preview.jpg'),preview(small,1600)[0])
    del image,coverage
    os.replace(image_path,out/'mosaic.tif');os.replace(coverage_path,out/'coverage.tif')
    metadata=dict(width=int(w),height=int(h),dtype=str(dtype),channels=channels,canvas_origin_xy=lo.tolist(),canvas_offset_xy=(-lo).tolist(),chunk_size=chunk_size,interpolation='OpenCV linear, constant outside; mask interpolation must be >=0.999',blending='source-boundary feather min(x+1,y+1,width-x,height-y), normalized over valid sources',intensity_correction='none',rounding='nearest then clip once to source integer range',coverage='number of valid contributing tiles, zero means unacquired gap')
    save_json(out/'render.json',metadata)
    save_json(out/'provenance.json',dict(tiles=tiles,positions=positions,meshes=[m.tolist() for m in meshes],render=metadata,forward='P+p+bilinear(mesh,p)',backward='12 fixed-point iterations, residual <0.01px',weights='reconstruct from maps, masks and declared feather rule'))
    return metadata
