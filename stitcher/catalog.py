"""Engine metadata and dependency checks; no model downloads on page load."""
from importlib.util import find_spec
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CATALOG = {}
def add(id, label, family, license, source, package=None, setup=''):
    CATALOG[id] = dict(id=id, label=label, family=family, license=license, source=source, package=package, setup=setup)
add('sift', 'SIFT + ratio matching', 'Classical', 'OpenCV Apache-2.0', 'https://github.com/opencv/opencv')
add('orb', 'ORB + ratio matching', 'Classical', 'OpenCV Apache-2.0', 'https://github.com/opencv/opencv')
for id, label in [('xfeat','XFeat'), ('xfeat-star','XFeat semi-dense'), ('xfeat-lighterglue','XFeat + LighterGlue')]:
    add(id, label, 'Sparse / semi-dense', 'Apache-2.0; check checkpoint terms', 'https://github.com/verlab/accelerated_features', 'xfeat', '.venv/bin/python setup_xfeat.py; install requirements-comparison.txt')
for kind in ('aliked','disk','sift'):
    add(kind+'-lightglue', kind.upper()+' + LightGlue', 'Sparse contextual', 'BSD-3-Clause + Apache-2.0' if kind == 'aliked' else 'Apache-2.0', 'https://github.com/cvg/LightGlue', 'lightglue', 'pip install git+https://github.com/cvg/LightGlue.git')
for kind in ('outdoor','indoor'):
    add('loftr-'+kind, 'LoFTR '+kind, 'Detector-free', 'Apache-2.0 code; verify checkpoint/data terms', 'https://github.com/zju3dv/LoFTR', 'kornia', 'pip install kornia')
for id, label in [('roma','RoMa outdoor'),('tiny-roma','Tiny RoMa')]:
    add(id, label, 'Dense', 'MIT + Apache-2.0 components; verify weights', 'https://github.com/Parskatt/RoMa', 'romatch', 'pip install romatch')
add('mast3r','MASt3R', 'Geometry-aware', 'Research only: CC BY-NC-SA 4.0 + dataset restrictions', 'https://github.com/naver/mast3r', 'mast3r', 'Follow upstream recursive clone and dependency setup; add repository to PYTHONPATH.')
add('matchanything-eloftr','MatchAnything · ELoFTR','Cross-modality / detector-free','Pinned source and checkpoint terms need review; see local LICENSE','https://zju3dv.github.io/MatchAnything/',setup='.venv/bin/python setup_matchanything.py')
add('superglue','SuperPoint + SuperGlue', 'Legacy baseline', 'Restricted / noncommercial; review upstream', 'https://github.com/magicleap/SuperGluePretrainedNetwork')

add('efficientloftr', 'EfficientLoFTR', 'Detector-free',
    'Pinned source and checkpoint terms need review; see local LICENSE',
    'https://github.com/zju3dv/EfficientLoFTR', setup='.venv/bin/python setup_efficientloftr.py')

# A family is accounted for without pretending a descriptor/refiner is an engine.
# None of these entries can reach model dispatch until an adapter is validated.
BLOCKED = [
    ('loma', 'LoMa', 'davnords/LoMa', 'B/B128/L/G/R checkpoint pins and an original-coordinate adapter are not validated.'),
    ('rdd', 'RDD', 'xtcpete/rdd', 'Corrected 2026 feature and compatible matcher checkpoint pair has not been pinned or tested.'),
    ('dedode', 'DeDoDe v2', 'Parskatt/DeDoDe', 'Detector/descriptor versions and compatible assignment require a validated pipeline.'),
    ('dad', 'DaD detector component', 'parskatt/dad', 'Detector only; a compatible descriptor and matcher must be validated first.'),
    ('roma-v2', 'RoMa v2', 'Parskatt/romav2', 'Separate adapter for overlap/precision output and device-memory smoke tests required.'),
    ('ufm', 'UFM', 'UniFlowMatch/UFM', 'Released large checkpoints need device-memory validation; unreleased Tiny weights are excluded.'),
    ('edm-efficient', 'Efficient Deep Feature Matching', 'chicleee/EDM', 'Pinned inference deployment, checkpoint and coordinate tests required.'),
    ('matchanything-roma', 'MatchAnything · RoMa', 'zju3dv/MatchAnything', 'RoMa backbone/checkpoint not provided by the ELoFTR setup; separate adapter required.'),
]
for id, label, repo in [
    ('alike','ALIKE','Shiaoming/ALIKE'), ('ripe','RIPE','fraunhoferhhi/RIPE'),
    ('ripe-plus','RIPE++','fraunhoferhhi/RIPEpp'), ('liftfeat','LiftFeat','lyp-deeplearning/LiftFeat'),
    ('silk','SiLK','facebookresearch/silk'), ('r2d2','R2D2','naver/r2d2'),
    ('fast-r2d2','Fast-R2D2','naver/r2d2'), ('d2-net','D2-Net','mihaidusmanu/d2-net'),
    ('aslfeat','ASLFeat','lzx551402/ASLFeat'), ('dalf','DALF','verlab/DALF_CVPR_2023'),
    ('zippypoint','ZippyPoint','menelaoskanakis/ZippyPoint'), ('lf-net','LF-Net','vcg-uvic/lf-net-release'),
    ('lift','LIFT','cvlab-epfl/tf-lift'), ('upal','UPAL points','francois141/upal'),
    ('dkm','DKM','Parskatt/DKM'), ('aspanformer','ASpanFormer','apple/ml-aspanformer'),
    ('matchformer','MatchFormer','InSAI-Lab/MatchFormer'), ('topicfm','TopicFM','TruongKhang/TopicFM'),
    ('topicfm-plus','TopicFM+','TruongKhang/TopicFM'), ('se2-loftr','SE2-LoFTR','georg-bn/se2-loftr'),
    ('casmtr','CasMTR','ewrfcas/CasMTR'), ('pats','PATS','zju3dv/pats'), ('astr','ASTR','ASTR2023/ASTR'),
    ('glu-net','GLU-Net','PruneTruong/DenseMatching'), ('gocor','GOCor pipeline','PruneTruong/DenseMatching'),
    ('pdc-net','PDC-Net','PruneTruong/DenseMatching'), ('pdc-net-plus','PDC-Net+','PruneTruong/DenseMatching'),
    ('minima','MINIMA','LSXI7/MINIMA'), ('gim','GIM','xuelunshen/gim'), ('xoftr','XoFTR','OnderT/XoFTR'),
    ('raft','RAFT','princeton-vl/RAFT'), ('gmflow','GMFlow','haofeixu/gmflow'),
    ('unimatch','UniMatch flow','autonomousvision/unimatch'), ('edm-equirectangular','Equirectangular EDM','jdk9405/EDM'),
    ('omniglue','OmniGlue','google-research/omniglue'), ('superpoint-lightglue','SuperPoint + LightGlue','cvg/LightGlue'),
    ('gluestick','GlueStick','cvg/GlueStick'),
]:
    reason = 'Pinned source/checkpoint, isolated dependency recipe and original-pixel inference tests are missing.'
    if id in ('minima','gim','gluestick'):
        reason = 'Concrete backbone/extractor/checkpoint combination has not been audited; SuperPoint dependency is unresolved.'
    elif id in ('raft','gmflow','unimatch'):
        reason = 'Occlusion and nonoverlap filtering must be validated before flow can constrain global geometry.'
    elif id == 'edm-equirectangular':
        reason = 'Specialist projection assumptions and geometry adapter are not validated for planar microscope tiles.'
    elif id in ('aslfeat','lf-net','lift'):
        reason = 'Legacy runtime requires an isolated worker and a tested coordinate protocol.'
    BLOCKED.append((id,label,repo,reason))
for id, label, url in [('sfd2','SFD2','https://arxiv.org/abs/2304.14845'),
                        ('clidd','CLIDD','https://arxiv.org/abs/2601.09230'),
                        ('redfeat','ReDFeat','https://arxiv.org/abs/2205.07439')]:
    BLOCKED.append((id,label,url,'Author release, exact inference checkpoint and dependency recipe have not been audited.'))
for id,label,repo in [
    ('hardnet','HardNet','DagnyT/hardnet'), ('hardnet-plus','HardNet++','DagnyT/hardnet'),
    ('sosnet','SOSNet','scape-research/SOSNet'), ('contextdesc','ContextDesc','lzx551402/contextdesc'),
    ('matcha','MATCHA','feixue94/matcha'), ('key-net','Key.Net hybrid detector','axelBarroso/Key.Net'),
    ('patch2pix','Patch2Pix refiner','GrumpyZhou/patch2pix'),
    ('keypt2subpx','Keypt2Subpx refiner','KimSinjeong/keypt2subpx'),
    ('rotation-steerers','Rotation steerers','georg-bn/rotation-steerers'),
    ('affine-steerers','Affine steerers','georg-bn/affine-steerers'),
]:
    BLOCKED.append((id,label,repo,'Component only: no compatible complete detector/descriptor/matcher/refiner pipeline has been validated.'))
for id,label,repo,reason in BLOCKED:
    add(id,label,'Experimental / unavailable','Unresolved; review exact source, dependencies and checkpoint terms',
        repo if repo.startswith('https:') else 'https://github.com/'+repo)
    CATALOG[id].update(implementation_status='blocked', blocking_reason=reason)

PAIRWISE = {'xfeat-star','loftr-outdoor','loftr-indoor','efficientloftr','matchanything-eloftr','roma','tiny-roma','mast3r'}
RGB = {'aliked-lightglue','disk-lightglue','roma','tiny-roma','mast3r'}
LOCAL_ASSETS = {
    'superglue': ['weights/superpoint_v1.pth','weights/superglue_outdoor.pth','weights/superglue_indoor.pth'],
    'efficientloftr': ['weights/eloftr_outdoor.ckpt','vendor/EfficientLoFTR/src/loftr/loftr.py'],
    'matchanything-eloftr': ['weights/matchanything_eloftr.ckpt','vendor/MatchAnything/src/loftr/loftr.py'],
}
PINS = {
    'efficientloftr': ('07e9c1401e71e9c556b1fda9d86d060af36ba176','0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878'),
    'matchanything-eloftr': ('6a7bcb589ec8da3a9e861e799122beaa5eba2193','b394fd368488e5e3136906daac825fadd3dd5376755dcdf5cb6ede0de20f6e7e'),
}

def option(kind, default, help, **constraints):
    return dict(type=kind, default=default, help=help, **constraints)

for id, entry in CATALOG.items():
    entry.setdefault('implementation_status', 'implemented')
    entry.setdefault('blocking_reason', None)
    entry['description'] = entry['label'] + ' · ' + entry['family']
    entry['source_revision'] = PINS.get(id, (None,None))[0]
    if id.startswith('xfeat'):
        entry['source_revision'] = 'e92685f57f8318b18725c5c8c0bd28c7fe188d9a'
    if id.startswith('loftr-'): entry['source_revision'] = 'kornia==0.8.3'
    if id in ('roma','tiny-roma'): entry['source_revision'] = 'romatch==0.1.2'
    if id.endswith('-lightglue') and entry['implementation_status'] != 'blocked':
        entry['source_revision'] = 'eb42fee2d71449efb0aa5c10549752b5d75384d8'
    entry['checkpoint'] = dict(id=id, sha256=PINS.get(id,(None,None))[1])
    if id == 'efficientloftr': entry['license'] = 'Project Registration License v1.0 (pinned source LICENSE); checkpoint terms separate'
    if id == 'matchanything-eloftr': entry['license'] = 'Apache-2.0 (pinned Space source LICENSE); checkpoint terms separate'
    entry['licenses'] = dict(code=entry['license'], checkpoint='Review upstream checkpoint terms', dependencies='Separate upstream terms', unresolved=True)
    entry['capabilities'] = dict(kind='pairwise' if id in PAIRWISE else 'sparse',
        color='rgb' if id in RGB else 'gray', reusable_features=id not in PAIRWISE,
        confidence=id not in ('xfeat','xfeat-star','mast3r'),
        devices=['cpu'] if id in ('sift','orb') else ['cpu','cuda'],
        incremental=entry['implementation_status'] != 'blocked',
        uses_superpoint=True if id in ('superglue','superpoint-lightglue','omniglue') else (None if id in ('minima','gim','gluestick') else False),
        teacher_superpoint=True if id == 'upal' else None, refiners=[], variants=[])
    schema = dict(
        device=option('enum','auto','Execution device; no silent fallback.',values=['auto',*entry['capabilities']['devices']]),
        feature_max_dim=option('integer',832 if id in ('efficientloftr','matchanything-eloftr') else 1024,'Maximum matching input dimension.',min=256,max=4096),
    )
    if id in PAIRWISE:
        schema['dense_sample_budget'] = option('integer',2048,'Spatial sample budget for pairwise correspondences.',min=128,max=16384)
    else:
        schema['max_keypoints'] = option('integer',1024,'Sparse feature budget per image.',min=128,max=8192)
    if id == 'superglue':
        schema.update(superglue_weights=option('enum','outdoor','SuperGlue training checkpoint.',values=['outdoor','indoor']),
            sinkhorn_iterations=option('integer',50,'SuperGlue assignment iterations.',min=1,max=200),
            match_threshold=option('number',.2,'SuperGlue match score threshold.',min=0,max=1))
        entry['capabilities']['variants'] = ['outdoor','indoor']
    if id in ('sift','orb'):
        schema['ratio_test'] = option('number',.75,'Nearest-neighbor descriptor ratio.',min=.1,max=.99)
    entry['options_schema'] = schema
    entry['presets'] = {name: {k: min(schema[k].get('max',v),v) for k,v in
        dict(feature_max_dim=min(res,832) if id in ('efficientloftr','matchanything-eloftr') else res,
             max_keypoints=budget,dense_sample_budget=budget,sinkhorn_iterations=iters).items() if k in schema}
        for name,res,budget,iters in [('fast',512,512,20),('balanced',1024,1024,50),('best',1600,2048,100)]}


def resolve_options(id, options=None, legacy=None):
    """Validate a versioned object, migrating only supported historical keys."""
    import math
    if id == 'superpoint+superglue': id = 'superglue'
    if id not in CATALOG:
        raise ValueError(f'Unknown engine: {id}')
    entry = CATALOG[id]
    if entry['implementation_status'] == 'blocked':
        raise ValueError(entry['blocking_reason'])
    if options is None: options = {}
    if not isinstance(options, dict) or type(options.get('version',1)) is not int or options.get('version',1) != 1:
        raise ValueError('engine_options must be an object with version 1')
    schema = entry['options_schema']
    unknown = set(options) - set(schema) - {'version'}
    if unknown: raise ValueError(f'Unsupported options for {id}: {", ".join(sorted(unknown))}')
    resolved = {'version':1}
    for key, spec in schema.items():
        value = options.get(key, (legacy or {}).get(key, spec['default']))
        if spec['type'] == 'enum':
            if value not in spec['values']: raise ValueError(f'Invalid {key}: {value}')
        else:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
            if spec['type'] == 'integer' and not isinstance(value,int): raise ValueError(f'{key} must be an integer')
            if not spec['min'] <= value <= spec['max']: raise ValueError(f'{key} is out of range')
        resolved[key] = value
    return resolved


def dependencies_present(entry):
    id, package = entry['id'], entry['package']
    packages = ['numpy','cv2','torch']
    if package == 'xfeat':
        if not (ROOT/'vendor/accelerated_features/hubconf.py').is_file(): return False
        packages += ['kornia']
    elif package: packages += [package]
    if id in ('efficientloftr','matchanything-eloftr'):
        packages += ['einops','kornia','yacs','loguru','PIL']
        if id == 'efficientloftr': packages += ['pytorch_lightning']
    try:
        return all(find_spec(p) is not None for p in packages)
    except (ImportError, ValueError, AttributeError):
        return False


def engines():
    from .verification import smoke_record
    result = []
    for id, entry in CATALOG.items():
        implemented = entry['implementation_status'] != 'blocked'
        deps = implemented and dependencies_present(entry)
        assets = LOCAL_ASSETS.get(id, [])
        local = all((ROOT/p).is_file() for p in assets)
        record = smoke_record(id) if implemented else None
        verified = bool(record and record.get('passed'))
        # Compatibility baselines are exercised by the repository's real-inference tests.
        available = deps and local and (verified or id in ('sift','orb','superglue'))
        status = (entry['blocking_reason'] if not implemented else
                  'Setup required' if not deps or not local else
                  'Verified in this environment' if verified else
                  'Compatibility baseline; smoke test available' if available else
                  'Installed; run inference verification before use')
        result.append({**entry, 'available':available, 'installed':deps and local,
            'implementation_status':'verified' if verified else entry['implementation_status'],
            'dependency_status':'present' if deps else 'missing',
            'weight_status':'verified by inference' if verified else ('local, unverified' if assets and local else 'missing' if not local else 'unverified; may download on verification'),
            'device_status':record.get('device') if record else 'untested',
            'verification':record, 'status':status})
    return result


def require_available(id, options=None, legacy=None, incremental=False):
    resolved = resolve_options(id, options, legacy)
    entry = next(e for e in engines() if e['id'] == id)
    if not entry['available']: raise ValueError(f'{entry["label"]}: {entry["status"]}. {entry["setup"]}')
    if incremental and not entry['capabilities']['incremental']:
        raise ValueError('This model supports batch grids and pair diagnostics. Incremental feature merging is not supported.')
    import torch
    actual_device = ('cuda' if torch.cuda.is_available() else 'cpu') if resolved['device']=='auto' else resolved['device']
    record = entry['verification']
    if id not in ('sift','orb','superglue') and record and record.get('device') != actual_device:
        raise ValueError(f'{id} has not passed verification on {actual_device}. Verify that device first.')
    if resolved['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select CPU explicitly.')
    return resolved


def validate_geometry(threshold, reference='middle', blend_levels=6):
    import math
    if not math.isfinite(threshold) or not 0 < threshold <= 20:
        raise ValueError('RANSAC threshold must be finite and between 0 and 20 pixels')
    if reference not in ('middle','first'): raise ValueError('Invalid reference image')
    if not 1 <= blend_levels <= 8: raise ValueError('Blend levels must be between 1 and 8')
