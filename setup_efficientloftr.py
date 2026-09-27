"""Install the pinned EfficientLoFTR source, inference dependencies and weights."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REVISION = '07e9c1401e71e9c556b1fda9d86d060af36ba176'
SHA256 = '0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878'


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-guided.txt')], check=True)
    repo = ROOT / 'vendor/EfficientLoFTR'
    if not repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/zju3dv/EfficientLoFTR.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Expected EfficientLoFTR revision {REVISION}; found {revision}. Use a separate checkout of the pinned revision.')
    # Upstream uses Kornia's old module path; this is the only source patch.
    source = repo / 'src/loftr/utils/fine_matching.py'
    source.write_text(source.read_text().replace('from kornia.utils.grid import create_meshgrid', 'from kornia.geometry import create_meshgrid'))
    path = ROOT / 'weights/eloftr_outdoor.ckpt'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        import gdown
        temporary = path.with_suffix('.download')
        gdown.download(id='1jFy2JbMKlIp82541TakhQPaoyB5qDeic', output=str(temporary), quiet=False)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('EfficientLoFTR checkpoint checksum mismatch; download not installed.')
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Installed EfficientLoFTR checkpoint checksum mismatch.')
    from stitcher.matching import build_engine
    engine = build_engine({'engine': 'efficientloftr'})
    print(f'EfficientLoFTR ready on {engine.device}. Start all web tools with ./start')


if __name__ == '__main__':
    main()
