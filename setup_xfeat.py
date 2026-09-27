"""Install the XFeat source/checkpoints validated by the shared catalog."""
from pathlib import Path
import subprocess
from stitcher.catalog import CATALOG

ROOT = Path(__file__).resolve().parent
REVISION = CATALOG['xfeat']['source_revision']


def main():
    repo = ROOT/'vendor/accelerated_features'
    if not repo.exists():
        repo.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout','https://github.com/verlab/accelerated_features.git',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'checkout',REVISION],check=True)
    actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f'XFeat requires {REVISION}; found {actual}. Prepare a separate pinned checkout; existing changes were not overwritten.')
    for name in ('xfeat.pt','xfeat-lighterglue.pt'):
        if not (repo/'weights'/name).is_file():
            raise RuntimeError(f'Missing XFeat checkpoint: {name}')
    print('XFeat source and checkpoints installed. Use model verification in /classic before running.')


if __name__=='__main__': main()
