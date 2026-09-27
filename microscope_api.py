"""Isolated CLI jobs for microscope mosaicing; paths come only from managed inputs."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import uuid
from fastapi import APIRouter,Form,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from microscopy.io import demo_manifest,save_json
from microscopy.pipeline import configuration

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'uploads/microscope'
router=APIRouter();jobs={};lock=threading.Lock()

@router.get('/microscope')
def page():return FileResponse(ROOT/'static/microscope.html')

@router.post('/api/microscope')
async def start(options:str=Form(...),manifest:UploadFile|None=File(None),files:list[UploadFile]=File(default=[])):
    try:
        opts=json.loads(options)
        cfg=configuration(opts.get('config'))
        with lock:
            if any(j.get('process') is None or j['process'].poll() is None for j in jobs.values() if j['status'] in ('preparing','running')):raise HTTPException(409,'A microscope job is already running')
            id=uuid.uuid4().hex;directory=RUNS/id;directory.mkdir(parents=True)
            jobs[id]={'status':'preparing','process':None}
        if manifest:
            data=await manifest.read(2*1024*1024+1)
            if len(data)>2*1024*1024:raise ValueError('Manifest must be under 2 MB')
            m=json.loads(data);uploads={}
            if len(files)>200:raise ValueError('Maximum 200 uploaded files')
            for f in files:
                name=f.filename or ''
                if not name or Path(name).name!=name or name in uploads:raise ValueError('Uploaded filenames must be unique plain filenames')
                payload=await f.read(100*1024*1024+1)
                if len(payload)>100*1024*1024:raise ValueError('Each image must be under 100 MB')
                dest=directory/'inputs'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(payload);uploads[name]=str(dest)
            for tile in m.get('tiles',[]):
                for key in ('path','mask'):
                    if key not in tile:continue
                    name=Path(tile[key]).name
                    if name not in uploads:raise ValueError('Missing uploaded file: '+name)
                    tile[key]=uploads[name]
        else:
            dataset=opts.get('dataset')
            if not isinstance(dataset,str) or Path(dataset).name!=dataset:raise ValueError('Choose a demo dataset')
            directory_demo=ROOT/'demo_content'/dataset
            if not (directory_demo/'positions.json').is_file():raise ValueError('Unknown demo dataset')
            m=demo_manifest(directory_demo)
            layout=opts.get('layout','20')
            if layout=='4':m['tiles']=[t for t in m['tiles'] if t['row']<2 and t['column']<2]
            elif layout=='20':m['tiles']=[t for t in m['tiles'] if t['row']<4 and t['column']<5]
            elif layout!='all':raise ValueError('Invalid demo layout')
        save_json(directory/'input-manifest.json',m);save_json(directory/'config.json',cfg)
        command='benchmark' if opts.get('benchmark') else 'run'
        args=[sys.executable,str(ROOT/'microscope_mosaic.py'),command,'--manifest',str(directory/'input-manifest.json'),'--config',str(directory/'config.json'),'--out',str(directory/'result')]
        with open(directory/'worker.log','w') as log:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        jobs[id].update(status='running',process=process)
        return {'id':id}
    except HTTPException:raise
    except Exception as exc:
        if 'id' in locals():jobs[id]['status']='failed'
        raise HTTPException(400,str(exc))

@router.get('/api/microscope/{id}')
def status(id:str):
    if not id.isalnum() or id not in jobs:raise HTTPException(404,'Unknown microscope job')
    job=jobs[id];directory=RUNS/id
    code=job['process'].poll() if job['process'] is not None else None
    if code is not None:job['status']='complete' if code==0 else 'failed'
    log=(directory/'worker.log').read_text(errors='replace')[-6000:] if (directory/'worker.log').exists() else ''
    reports=[]
    for p in sorted((directory/'result').glob('*/summary.json')):
        report=json.loads(p.read_text());report['prefix']=str(p.parent.relative_to(directory));reports.append(report)
    artifacts=[str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.html','.jpg','.tif') and '/sources/' not in str(p) and '/inputs/' not in str(p) and 'refinement_cache' not in str(p) and '.partial.' not in p.name]
    return dict(id=id,status=job['status'],log=log,reports=reports,artifacts=artifacts)

@router.get('/api/microscope/{id}/files/{path:path}')
def artifact(id:str,path:str):
    if not id.isalnum():raise HTTPException(404)
    root=(RUNS/id).resolve();target=(root/path).resolve()
    if not target.is_relative_to(root) or not target.is_file():raise HTTPException(404)
    return FileResponse(target)
