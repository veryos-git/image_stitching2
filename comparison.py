"""Comparison routes, with serial model execution and independently reported errors."""
import asyncio
import gc
import json
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from stitcher import PanoramaStitcher, StitchConfig, ProgressReporter
from stitcher.catalog import engines, CATALOG, require_available

router = APIRouter()
ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'uploads/comparisons'
DEMO = ROOT/'demo_content'
records = {}
lock = threading.Lock()
compute_lock = threading.Lock()


def demo_sets():
    sets = []
    for directory in sorted(DEMO.iterdir()) if DEMO.exists() else []:
        files = sorted(directory.glob('tile_*.png'))
        # Snake order maintains adjacent overlap at row boundaries.
        rows = {}
        for f in files:
            rows.setdefault(f.name.split('_c')[0], []).append(f)
        ordered = [f.name for i, row in enumerate(rows.values()) for f in (row if i % 2 == 0 else row[::-1])]
        if ordered:
            sets.append(dict(id=directory.name, files=ordered))
    return sets

@router.get('/api/engines')
def catalog():
    return {'engines': engines(), 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

@router.get('/api/demos')
def demos():
    return {'datasets': demo_sets()}

@router.get('/api/demos/{dataset}/{filename}')
def demo_image(dataset: str, filename: str):
    valid = next((d for d in demo_sets() if d['id'] == dataset), None)
    if not valid or filename not in valid['files']:
        raise HTTPException(404, 'Unknown demo image')
    return FileResponse(DEMO/dataset/filename)


def update(id, **values):
    with lock:
        records[id].update(values)


def run(id, paths, selected, settings):
    with compute_lock:
        update(id, status='running')
        for index, engine in enumerate(selected):
            update(id, current=engine, progress=f'Loading {CATALOG[engine]["label"]}')
            row = dict(engine=engine, status='running', matches=[], inliers=[], keypoints=[], stages={})
            with lock:
                records[id]['results'].append(row)
            start = time.perf_counter()
            stage_start = {}
            def emit(event):
                with lock:
                    if event['type'] == 'step':
                        marker = event.get('marker')
                        records[id]['progress'] = event.get('message', marker)
                        if event.get('status') == 'running':
                            stage_start.setdefault(marker, time.perf_counter())
                        if event.get('status') == 'done':
                            row['stages'][marker] = round(time.perf_counter() - stage_start.get(marker, start), 3)
                            detail = event.get('detail', {})
                            for src, dst in [('matches_per_pair','matches'),('inliers_per_pair','inliers'),('per_image','keypoints')]:
                                if src in detail: row[dst] = detail[src]
            stitcher = None
            try:
                cv2.setRNGSeed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                per_engine = dict(settings)
                options = per_engine.pop('engine_options',{}).get(engine,{})
                cfg = StitchConfig(engine=engine,engine_options=options,weights_dir=str(ROOT/'weights'),viz=False,max_output_dim=4096,**per_engine)
                row['engine_options'] = cfg.engine_options
                stitcher = PanoramaStitcher(cfg, ProgressReporter(emit=emit))
                result = stitcher.stitch_paths(paths)
                dest = RUNS/id/(engine+'.jpg')
                if not cv2.imwrite(str(dest), result.panorama):
                    raise RuntimeError('Could not save panorama')
                with lock:
                    row['warning'] = 'Low geometric agreement: inspect this panorama carefully.' if any(i / max(m, 1) < 0.15 for i, m in zip(row['inliers'], row['matches'])) else None
                    row['graph'] = result.stats.get('graph')
                    if row['graph']:
                        row['warning'] = None
                    if row['graph'] and row['graph']['excluded']:
                        row['warning'] = f"Partial panorama: excluded input images {row['graph']['excluded']} (no reliable connection to the largest group)."
                    row.update(status='complete', output=result.stats['output'], image=f'/api/comparisons/{id}/images/{engine}', stats=result.stats)
                del result
            except Exception as exc:
                with lock:
                    row.update(status='failed', error=str(exc), graph=getattr(exc, 'graph', None))
            finally:
                with lock:
                    row['elapsed'] = round(time.perf_counter()-start, 3)
                del stitcher
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        with lock:
            records[id].update(status='complete', current=None, progress='Comparison complete')
            (RUNS/id/'report.json').write_text(json.dumps(records[id], indent=2))

@router.post('/api/comparisons')
async def compare(options: str = Form(...), files: list[UploadFile] = File(default=[])):
    try:
        opts = json.loads(options)
        if not isinstance(opts, dict) or not isinstance(opts.get('engines'), list) or not all(isinstance(e, str) for e in opts['engines']):
            raise ValueError('Options must include a list of engine names')
        selected = list(dict.fromkeys(opts['engines']))
        if not selected or any(e not in CATALOG for e in selected): raise ValueError('Choose valid engines')
        settings = dict(alignment=opts.get('alignment','homography'), feature_max_dim=int(opts.get('feature_max_dim',1024)), max_keypoints=int(opts.get('max_keypoints',2048)), crop=bool(opts.get('crop',False)), pairing=opts.get('pairing','sequential'), disconnected=opts.get('disconnected','largest'))
        settings['engine_options'] = {id:require_available(id,opts.get('engine_options',{}).get(id),settings) for id in selected}
        if settings['alignment'] not in ('homography','translation'): raise ValueError('Invalid alignment')
        if settings['pairing'] not in ('sequential','unordered') or settings['disconnected'] not in ('largest','reject'): raise ValueError('Invalid stitching mode')
        if not 256 <= settings['feature_max_dim'] <= 1600 or not 128 <= settings['max_keypoints'] <= 8192: raise ValueError('Settings out of range')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(400, str(exc))
    with lock:
        if any(r['status'] in ('queued','running') for r in records.values()):
            raise HTTPException(409, 'A comparison is already running; wait for it to finish.')
        id = uuid.uuid4().hex
        records[id] = dict(id=id, status='queued', current=None, progress='Queued', results=[], settings=settings, engines=selected, device='cuda' if torch.cuda.is_available() else 'cpu', versions={'torch':torch.__version__, 'opencv':cv2.__version__, 'numpy':np.__version__}, seed=0)
    directory = RUNS/id
    directory.mkdir(parents=True)
    try:
        paths = []
        names = []
        if files:
            if not 2 <= len(files) <= 36: raise ValueError('Upload 2–36 images')
            for i, file in enumerate(files):
                data = await file.read(30*1024*1024+1)
                if len(data) > 30*1024*1024: raise ValueError('Each image must be under 30 MB')
                image = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None: raise ValueError(f'Cannot decode {file.filename}')
                if image.shape[0]*image.shape[1] > 25_000_000: raise ValueError('Images must be under 25 megapixels')
                dest = directory/f'input_{i:03d}.png'
                await asyncio.to_thread(cv2.imwrite, str(dest), image)
                paths.append(str(dest)); names.append(file.filename)
        else:
            dataset = next((d for d in demo_sets() if d['id'] == opts.get('dataset')), None)
            if not dataset: raise ValueError('Choose a demo dataset')
            names = opts.get('demo_files', dataset['files'][:3])
            if not 2 <= len(names) <= 36 or len(set(names)) != len(names) or any(n not in dataset['files'] for n in names): raise ValueError('Choose 2–36 distinct demo tiles')
            paths = [str(DEMO/dataset['id']/n) for n in names]
        update(id, inputs=names, sample_seed=opts.get('sample_seed'), input_test=opts.get('input_test','uploads' if files else 'demo'))
    except Exception as exc:
        update(id, status='failed', progress=str(exc))
        raise HTTPException(400, str(exc))
    threading.Thread(target=run, args=(id, paths, selected, settings), daemon=True).start()
    return {'id': id}

@router.get('/api/comparisons/{id}')
def status(id: str):
    with lock:
        if id not in records: raise HTTPException(404, 'Unknown comparison')
        return json.loads(json.dumps(records[id]))

@router.get('/api/comparisons/{id}/report')
def report(id: str):
    path = RUNS/id/'report.json'
    if not id.isalnum() or not path.is_file(): raise HTTPException(404, 'Report not ready')
    return FileResponse(path, filename='comparison.json')

@router.get('/api/comparisons/{id}/images/{engine}')
def output(id: str, engine: str):
    if not id.isalnum() or engine not in CATALOG: raise HTTPException(404)
    path = RUNS/id/(engine+'.jpg')
    if not path.exists(): raise HTTPException(404)
    return FileResponse(path)


@router.post('/api/engines/{engine}/verify')
def verify(engine: str, options: dict = None):
    from stitcher.verification import verify_engine
    try:
        with compute_lock:
            record = verify_engine(engine, options)
        return record
    except ValueError as exc:
        raise HTTPException(400,str(exc))


@router.get('/api/engines/{engine}/verification-preview')
def verification_preview(engine: str):
    from stitcher.verification import RECORDS
    if engine not in CATALOG or not (RECORDS/(engine+'.jpg')).is_file():
        raise HTTPException(404,'No verification artifact')
    return FileResponse(RECORDS/(engine+'.jpg'))
