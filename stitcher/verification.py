"""Local inference evidence, invalidated by runtime/source/checkpoint changes."""
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / '.model_cache/verification'


def environment_key(id):
    from .catalog import LOCAL_ASSETS, CATALOG
    packages = {}
    for name in ('torch','torchvision','numpy','opencv-python-headless','kornia','lightglue','romatch'):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    files = {}
    def track(path):
        if not path.is_file(): return
        stat = path.stat()
        # Setup scripts may rewrite identical source. Content identity prevents
        # that from invalidating inference evidence or unrelated model records.
        value = hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == '.py' or path.name == 'SOURCE_REVISION' else [stat.st_size,stat.st_mtime_ns]
        files[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = value
    for path in [*sorted((ROOT/'stitcher').glob('*.py')), *(ROOT/p for p in LOCAL_ASSETS.get(id,[]))]:
        track(path)
    folders = []
    if id.startswith('xfeat') or id=='tiny-roma': folders.append('accelerated_features')
    if id=='efficientloftr': folders.append('EfficientLoFTR')
    if id=='matchanything-eloftr': folders.append('MatchAnything')
    for folder in folders:
        repo = ROOT/'vendor'/folder
        for path in sorted(repo.rglob('*')) if repo.exists() else []:
            if path.suffix in ('.py','.pt','.pth','.ckpt') or path.name=='SOURCE_REVISION': track(path)
    cache_names = {
        'aliked-lightglue':['aliked-n16.pth','aliked_lightglue_v0-1_arxiv.pth'],
        'disk-lightglue':['depth-save.pth','disk_lightglue_v0-1_arxiv.pth'],
        'sift-lightglue':['sift_lightglue_v0-1_arxiv.pth'],
        'loftr-outdoor':['loftr_outdoor.ckpt'], 'loftr-indoor':['loftr_indoor.ckpt'],
        'roma':['roma_outdoor.pth','dinov2_vitl14_pretrain.pth'],
        'tiny-roma':['tiny_roma_v1_outdoor.pth'],
    }
    import os
    cache = Path(os.environ.get('TORCH_HOME',ROOT/'.model_cache'))/'hub/checkpoints'
    for name in cache_names.get(id,[]): track(cache/name)
    data = dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                files=files, source_revision=CATALOG[id]['source_revision'])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(), data


def smoke_record(id):
    path = RECORDS/(id+'.json')
    if not path.is_file(): return None
    try:
        record = json.loads(path.read_text())
        if record['environment_key'] != environment_key(id)[0]: return None
        return record
    except (ValueError,KeyError,OSError): return None


def verify_engine(id, options=None):
    import cv2
    import numpy as np
    import torch
    from .catalog import CATALOG, resolve_options, dependencies_present
    from .matching import build_engine
    from .contract import prepare_image, match_pair
    from .unordered import reliable_edge
    from .utils import draw_matches
    resolved = resolve_options(id, options)
    if not dependencies_present(CATALOG[id]):
        raise ValueError('Missing dependencies. '+CATALOG[id]['setup'])
    RECORDS.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    record = dict(engine=id, options=resolved, passed=False, timestamp=time.time())
    try:
        torch.manual_seed(19)
        np.random.seed(19)
        model = build_engine(dict(engine=id, engine_options=resolved, weights_dir=str(ROOT/'weights')))
        rng = np.random.default_rng(19)
        scene = np.full((320,600,3),225,np.uint8)
        for _ in range(800):
            x,y = rng.integers([0,0],[600,320])
            cv2.circle(scene,(int(x),int(y)),int(rng.integers(2,9)),tuple(int(v) for v in rng.integers(0,220,3)),-1)
        for x in range(0,600,80):
            cv2.putText(scene,str(x),(x,160),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        fixture = ROOT/'demo_content/scan_2026-09-17_010421/tile_r00_c01.png'
        if fixture.is_file():
            scene = cv2.resize(cv2.imread(str(fixture)),(600,320),interpolation=cv2.INTER_AREA)
            record['fixture'] = dict(path=str(fixture.relative_to(ROOT)),sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                method='unequal crops of one resized real microscope tile; known translation (-120,-13)')
        else:
            record['fixture'] = {'method':'deterministic structured synthetic landmarks'}
        a = prepare_image(model,scene[:,:400],'known-left')
        b = prepare_image(model,scene[13:300,120:],'synthetic-right')
        matches = match_pair(model,a,b)
        H,count,reason = reliable_edge(matches.points0,matches.points1,a.shape,b.shape,3,'translation')
        error = None if H is None else float(np.linalg.norm(H[:2,2]-[-120,-13]))
        negative = prepare_image(model,rng.integers(0,256,(287,480,3),dtype=np.uint8),'independent-negative')
        nm = match_pair(model,a,negative)
        nh,_,_ = reliable_edge(nm.points0,nm.points1,a.shape,negative.shape,3,'translation')
        cv2.imwrite(str(RECORDS/(id+'.jpg')),draw_matches(a.small_gray,b.small_gray,
            matches.points0*a.scale,matches.points1*b.scale,matches.scores,max_draw=500))
        # Record exact loaded tensor identity even for upstream auto-downloaded weights.
        checkpoint = hashlib.sha256()
        for attr in ('model','matcher','superpoint','superglue'):
            module = getattr(model,attr,None)
            if not hasattr(module,'state_dict'): continue
            for key,tensor in sorted(module.state_dict().items()):
                checkpoint.update((attr+'.'+key).encode())
                checkpoint.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        record.update(passed=H is not None and error <= 2 and nh is None,
            matches=len(matches.points0), inliers=int(count), translation_error_px=error,
            negative_matches=len(nm.points0), negative_accepted=nh is not None,
            tolerance_px=2, device=str(getattr(model,'device','cpu')), loaded_state_sha256=checkpoint.hexdigest(),
            reason=reason or ('Negative pair falsely accepted' if nh is not None else None),
            artifact=f'/api/engines/{id}/verification-preview')
        if error is not None and error > 2: record['reason'] = 'Recovered translation exceeds 2 px tolerance'
    except Exception as exc:
        record['reason'] = str(exc)
    record['elapsed'] = time.perf_counter()-started
    record['environment_key'], record['environment'] = environment_key(id)
    target = RECORDS/(id+'.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2))
    temp.replace(target)
    return record
