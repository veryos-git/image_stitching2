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
    add(id, label, 'Sparse / semi-dense', 'Apache-2.0; check checkpoint terms', 'https://github.com/verlab/accelerated_features', 'xfeat', 'git clone https://github.com/verlab/accelerated_features vendor/accelerated_features; pip install kornia')
for kind in ('aliked','disk','sift'):
    add(kind+'-lightglue', kind.upper()+' + LightGlue', 'Sparse contextual', 'BSD-3-Clause + Apache-2.0' if kind == 'aliked' else 'Apache-2.0', 'https://github.com/cvg/LightGlue', 'lightglue', 'pip install git+https://github.com/cvg/LightGlue.git')
for kind in ('outdoor','indoor'):
    add('loftr-'+kind, 'LoFTR '+kind, 'Detector-free', 'Apache-2.0 code; verify checkpoint/data terms', 'https://github.com/zju3dv/LoFTR', 'kornia', 'pip install kornia')
for id, label in [('roma','RoMa outdoor'),('tiny-roma','Tiny RoMa')]:
    add(id, label, 'Dense', 'MIT + Apache-2.0 components; verify weights', 'https://github.com/Parskatt/RoMa', 'romatch', 'pip install romatch')
add('mast3r','MASt3R', 'Geometry-aware', 'Research only: CC BY-NC-SA 4.0 + dataset restrictions', 'https://github.com/naver/mast3r', 'mast3r', 'Follow upstream recursive clone and dependency setup; add repository to PYTHONPATH.')
add('matchanything-eloftr','MatchAnything · ELoFTR','Cross-modality / detector-free','Apache-2.0 source; see upstream checkpoint terms','https://zju3dv.github.io/MatchAnything/',setup='.venv/bin/python setup_matchanything.py')
add('superglue','SuperPoint + SuperGlue', 'Legacy baseline', 'Restricted / noncommercial; review upstream', 'https://github.com/magicleap/SuperGluePretrainedNetwork')

def engines():
    result = []
    for id, entry in CATALOG.items():
        package = entry['package']
        ready = True
        if package == 'xfeat':
            ready = (ROOT / 'vendor/accelerated_features/hubconf.py').is_file()
            if id == 'xfeat-lighterglue':
                ready = ready and find_spec('kornia') is not None
        elif package:
            ready = find_spec(package) is not None
            if id == 'tiny-roma':
                ready = ready and (ROOT / 'vendor/accelerated_features/hubconf.py').is_file()
        elif id == 'matchanything-eloftr':
            ready = (ROOT/'vendor/MatchAnything/src/loftr/loftr.py').is_file() and (ROOT/'weights/matchanything_eloftr.ckpt').is_file() and all(find_spec(p) is not None for p in ('einops','kornia','yacs','loguru','PIL'))
        elif id == 'superglue':
            ready = all((ROOT/'weights'/f).exists() for f in ('superpoint_v1.pth','superglue_outdoor.pth'))
        result.append({**entry, 'available': ready, 'status': 'Dependencies present; weights load on first run' if ready and package else ('Ready' if ready else 'Setup required')})
    return result
