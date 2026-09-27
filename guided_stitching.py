"""User-directed incremental stitching with transactional map updates."""
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from stitcher.matching import build_engine
from stitcher.translation import reliable_translation
from stitcher import blender

ROOT=Path(__file__).resolve().parent
STORE=ROOT/'uploads/guided'
router=APIRouter()
sessions={}


class GrowingMap:
    """Match originals to originals; preserve the existing mosaic on failed additions."""
    def __init__(self, engine='efficientloftr', edge_filter=False):
        if engine != 'efficientloftr':
            raise ValueError('Grow a mosaic only supports EfficientLoFTR learned matching.')
        self.engine=build_engine({'engine':'efficientloftr','feature_max_dim':832,'edge_filter':edge_filter})
        self.placed=[]
        self.image=None
        self.mask=None
        self.weight=None
        self.origin=np.zeros(2,dtype=np.int64)

    @torch.inference_mode()
    def features(self,image):return self.engine.extract(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))

    def seed(self,image,index):
        if max(image.shape[:2])>16000:raise ValueError('The main image exceeds the 16,000-pixel side limit.')
        fs=self.features(image)
        self.image=image.copy();self.mask=np.full(image.shape[:2],255,np.uint8)
        self.weight=blender.feather_mask(self.mask)
        self.placed=[dict(index=index,features=fs,offset=np.zeros(2,dtype=np.int64))]
        return {'width':image.shape[1],'height':image.shape[0]}

    @torch.inference_mode()
    def propose(self,image,index):
        fs=self.features(image);candidates=[]
        for reference in self.placed:
            a,b,_=self.engine.match(fs,reference['features'])
            offset,inliers=reliable_translation(a,b,fs.shape,reference['features'].shape)
            if offset is not None:
                candidates.append((inliers,reference['offset']+offset,reference['index'],len(a)))
        if not candidates:raise ValueError('No reliable translation-only overlap with the current mosaic. Images must have the same scale and orientation. Try a neighboring image first.')
        candidates.sort(key=lambda c:-c[0]);inliers,offset,reference,matches=candidates[0]
        h,w=image.shape[:2]
        corners=np.array([offset,offset+[w,h]])
        # Independently supported x/y placements must agree before committing.
        for support,other,_,_ in candidates[1:]:
            if support>=max(12,inliers*.5) and np.linalg.norm(offset-other)>12:
                raise ValueError('Matching images disagree about translation. The mosaic was not changed.')
        mh,mw=self.image.shape[:2]
        lo=np.minimum(corners.min(0),self.origin);hi=np.maximum(corners.max(0),self.origin+[mw,mh])
        nw,nh=map(int,hi-lo)
        if max(nw,nh)>16000 or nw*nh>40_000_000:raise ValueError('This addition exceeds the 40-megapixel / 16,000-pixel canvas limit. Use smaller images.')
        dx,dy=np.round(self.origin-lo).astype(int)
        oldmask=np.zeros((nh,nw),np.uint8);oldmask[dy:dy+mh,dx:dx+mw]=self.mask
        x,y=map(int,offset-lo)
        placed_image=np.zeros((nh,nw,3),np.uint8)
        placed_image[y:y+h,x:x+w]=image
        mask=np.zeros((nh,nw),np.uint8);mask[y:y+h,x:x+w]=255
        added=int(np.count_nonzero((mask>0)&(oldmask==0)))
        overlapping=int(np.count_nonzero((mask>0)&(oldmask>0)))
        if overlapping<100:raise ValueError('Placement has insufficient overlap with the mosaic.')
        if added<max(100,int(.001*np.count_nonzero(self.mask))):
            raise ValueError('This image matches but adds no meaningful new coverage. It remains available; the mosaic was not changed.')
        canvas=np.zeros((nh,nw,3),np.uint8);canvas[dy:dy+mh,dx:dx+mw]=self.image
        weights=np.zeros((nh,nw),np.float32);weights[dy:dy+mh,dx:dx+mw]=self.weight
        incoming=blender.feather_mask(mask)
        total=weights+incoming
        result=np.clip((canvas.astype(np.float32)*weights[...,None]+placed_image.astype(np.float32)*incoming[...,None])/np.maximum(total[...,None],1e-6),0,255).astype(np.uint8)
        result[(mask>0)&(oldmask==0)]=placed_image[(mask>0)&(oldmask==0)]
        result[(oldmask>0)&(mask==0)]=canvas[(oldmask>0)&(mask==0)]
        return dict(image=result,mask=((oldmask>0)|(mask>0)).astype(np.uint8)*255,weight=total,origin=lo,placed=self.placed+[dict(index=index,features=fs,offset=offset)],metrics=dict(inliers=inliers,matches=matches,matched_image=reference,translation_x=int(offset[0]),translation_y=int(offset[1]),added_pixels=added,width=nw,height=nh))

    def commit(self,proposal):
        for field in ('image','mask','weight','origin','placed'):setattr(self,field,proposal[field])


class Session:
    def __init__(self,id,items,engine,edge_filter=False):
        self.id=id;self.items=items;self.engine=engine;self.edge_filter=edge_filter;self.map=None
        self.lock=threading.Lock();self.version=0;self.history=[];self.message='Choose the image that will become your main image.'
    def state(self):
        used=[p['index'] for p in self.map.placed] if self.map else []
        return dict(id=self.id,items=[dict(index=i,name=t['name'],status='stitched' if i in used else ('retry' if t.get('error') else 'available'),error=t.get('error'),thumbnail=f'/api/guided/{self.id}/images/{i}?thumbnail=1') for i,t in enumerate(self.items)],used=used,version=self.version,engine=self.engine,alignment='translation',edge_filter=self.edge_filter,message=self.message,history=self.history,complete=len(used)==len(self.items),width=self.map.image.shape[1] if self.map else 0,height=self.map.image.shape[0] if self.map else 0,mosaic=f'/api/guided/{self.id}/mosaic/{self.version}.png' if self.map else None)


def get_session(id):
    if id not in sessions:raise HTTPException(404,'Session unavailable. Upload your image set again; sessions last until the server restarts.')
    return sessions[id]


def save_mosaic(session,image,version):
    path=STORE/session.id/f'mosaic-{version}.png'
    if not cv2.imwrite(str(path),image):raise RuntimeError('Could not save the mosaic')

@router.get('/guided')
def page():return FileResponse(ROOT/'static/guided.html')

@router.post('/api/guided')
def upload(files:list[UploadFile]=File(...),engine:str=Form('efficientloftr'),edge_filter:bool=Form(False)):
    if engine != 'efficientloftr':raise HTTPException(400,'Grow a mosaic only supports EfficientLoFTR learned matching.')
    if not 2<=len(files)<=60:raise HTTPException(400,'Upload 2–60 images')
    id=uuid.uuid4().hex;directory=STORE/id;directory.mkdir(parents=True)
    items=[];total=0
    for i,f in enumerate(files):
        data=f.file.read(30*1024*1024+1);total+=len(data)
        if len(data)>30*1024*1024 or total>500*1024*1024:raise HTTPException(400,'Limit: 30 MB per image and 500 MB total')
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise HTTPException(400,f'Cannot read {f.filename}')
        if image.shape[0]*image.shape[1]>20_000_000:raise HTTPException(400,'Each image must be at most 20 megapixels')
        path=directory/f'input-{i}.png'
        if not cv2.imwrite(str(path),image):raise HTTPException(500,'Could not save uploaded image')
        h,w=image.shape[:2];small=cv2.resize(image,(max(1,round(w*min(1,220/max(h,w)))),max(1,round(h*min(1,220/max(h,w))))))
        cv2.imwrite(str(directory/f'thumb-{i}.jpg'),small)
        items.append(dict(name=f.filename or f'Image {i+1}',path=str(path)))
    session=Session(id,items,engine,edge_filter);sessions[id]=session
    return session.state()

@router.get('/api/guided/{id}')
def status(id:str):
    session=get_session(id)
    if not session.lock.acquire(blocking=False):raise HTTPException(409,'An image is being stitched. Please wait.')
    try:return session.state()
    finally:session.lock.release()

@router.post('/api/guided/{id}/select/{index}')
def select(id:str,index:int):
    session=get_session(id)
    if not 0<=index<len(session.items):raise HTTPException(404,'Unknown image')
    if not session.lock.acquire(blocking=False):raise HTTPException(409,'An image is already being stitched')
    try:
        if session.map and any(p['index']==index for p in session.map.placed):raise HTTPException(409,'This image is already in the mosaic')
        tick=time.perf_counter();image=cv2.imread(session.items[index]['path'])
        if image is None:raise HTTPException(500,'The uploaded image is unavailable')
        try:
            if session.map is None:
                candidate=GrowingMap(session.engine,session.edge_filter);metrics=candidate.seed(image,index)
                save_mosaic(session,candidate.image,session.version+1)
                session.map=candidate
                message=f'Main image selected: {session.items[index]["name"]}. Choose an overlapping image to add.'
            else:
                proposal=session.map.propose(image,index)
                save_mosaic(session,proposal['image'],session.version+1)
                session.map.commit(proposal);metrics=proposal['metrics']
                message=f'Added {session.items[index]["name"]}. The mosaic gained {metrics["added_pixels"]:,} pixels of coverage.'
            session.version+=1;session.items[index].pop('error',None);ok=True
        except Exception as exc:
            message=str(exc);session.items[index]['error']=message;metrics={};ok=False
        session.message=message
        session.history.append(dict(index=index,name=session.items[index]['name'],ok=ok,message=message,seconds=round(time.perf_counter()-tick,3),**metrics))
        return {'ok':ok,**session.state()}
    finally:session.lock.release()

@router.post('/api/guided/{id}/reset')
def reset(id:str):
    session=get_session(id)
    if not session.lock.acquire(blocking=False):raise HTTPException(409,'Wait for the current addition to finish')
    try:
        session.map=None;session.history=[]
        for item in session.items:item.pop('error',None)
        session.message='Start over: choose a main image from the same uploads.'
        return session.state()
    finally:session.lock.release()

@router.get('/api/guided/{id}/images/{index}')
def input_image(id:str,index:int,thumbnail:bool=False):
    session=get_session(id)
    if not 0<=index<len(session.items):raise HTTPException(404)
    return FileResponse(STORE/id/(f'thumb-{index}.jpg' if thumbnail else f'input-{index}.png'))

@router.get('/api/guided/{id}/mosaic/{version}.png')
def mosaic(id:str,version:int):
    session=get_session(id);path=STORE/id/f'mosaic-{version}.png'
    if version<1 or not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type='image/png')
