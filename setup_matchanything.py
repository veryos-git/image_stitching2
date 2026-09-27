"""Install MatchAnything ELoFTR from the authors' published Space and weights."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
REVISION = '6a7bcb589ec8da3a9e861e799122beaa5eba2193'
SPACE = 'LittleFrog/MatchAnything'
PREFIX = 'imcui/third_party/MatchAnything/'
SHA256 = 'b394fd368488e5e3136906daac825fadd3dd5376755dcdf5cb6ede0de20f6e7e'


def main():
    subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-matchanything.txt')],check=True)
    import requests
    import gdown
    response = requests.get(f'https://huggingface.co/api/spaces/{SPACE}/revision/{REVISION}',timeout=60)
    response.raise_for_status()
    paths = [f['rfilename'] for f in response.json()['siblings'] if f['rfilename'].startswith(PREFIX) and (f['rfilename'][len(PREFIX):].startswith(('src/loftr/','src/config/','configs/models/')) or f['rfilename'][len(PREFIX):] in ('src/__init__.py','LICENSE','README.md'))]
    repo = ROOT/'vendor/MatchAnything'
    def fetch(path):
        result = requests.get(f'https://huggingface.co/spaces/{SPACE}/resolve/{REVISION}/{path}',timeout=60)
        result.raise_for_status()
        dest = repo/path[len(PREFIX):]
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(result.content)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch,paths))
    # Compatibility-only patches: modern Kornia and an isolated module namespace.
    p = repo/'src/loftr/utils/fine_matching.py'
    p.write_text(p.read_text().replace('from kornia.utils.grid import create_meshgrid','from kornia.geometry import create_meshgrid'))
    p = repo/'configs/models/eloftr_model.py'
    p.write_text(p.read_text().replace('from src.config.default import _CN as cfg','from vendor.MatchAnything.src.config.default import _CN as cfg'))
    checkpoint = ROOT/'weights/matchanything_eloftr.ckpt'
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'weights.zip'
            gdown.download(id='12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d',output=str(archive),quiet=False)
            with zipfile.ZipFile(archive) as z:
                data = z.read('weights/matchanything_eloftr.ckpt')
            if hashlib.sha256(data).hexdigest()!=SHA256:
                raise RuntimeError('MatchAnything checkpoint checksum mismatch.')
            temporary = checkpoint.with_suffix('.download')
            temporary.write_bytes(data)
            temporary.replace(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SHA256:
        raise RuntimeError('Installed MatchAnything checkpoint checksum mismatch.')
    (repo/'SOURCE_REVISION').write_text(REVISION+'\n')
    from stitcher.matching import build_engine
    model = build_engine({'engine':'matchanything-eloftr','feature_max_dim':832})
    print(f'MatchAnything ELoFTR ready on {model.device}. Start with ./start and open /compare.')


if __name__=='__main__':
    main()
