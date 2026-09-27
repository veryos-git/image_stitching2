#!/usr/bin/env python3
"""Explicit inference verification; never run during catalog/page loading."""
import argparse
import json
from pathlib import Path
from stitcher.catalog import CATALOG
from stitcher.verification import verify_engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engines',nargs='+',choices=list(CATALOG))
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    records = [verify_engine(id,{'version':1,'device':args.device}) for id in args.engines]
    text = json.dumps(records,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(json.dumps([{k:r.get(k) for k in ('engine','passed','reason','elapsed','translation_error_px','negative_accepted')} for r in records],indent=2))
    raise SystemExit(0 if all(r['passed'] for r in records) else 1)
