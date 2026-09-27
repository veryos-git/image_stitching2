#!/usr/bin/env python3
"""Independent microscope CLI; identity baseline does not import PyTorch."""
import argparse
import json
from pathlib import Path
import sys
from microscopy.io import validate,save_json,demo_manifest
from microscopy.pipeline import configuration,register,render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','register','render','refine','run','benchmark','demo-manifest'])
    parser.add_argument('--manifest');parser.add_argument('--config');parser.add_argument('--run');parser.add_argument('--out',required=True)
    parser.add_argument('--dataset',help='JSON with slides: [{manifest, split}] for paired benchmarks');parser.add_argument('--matrix',help='JSON/YAML configurations for complete-pipeline comparison');parser.add_argument('--demo-dir');parser.add_argument('--refiner',choices=['identity','farneback','pixelstitch'],default=None)
    parser.add_argument('--refiners',default='identity,farneback,pixelstitch')
    args=parser.parse_args()
    def emit(s):print(s,flush=True)
    cfg=None
    if args.config:
        import yaml
        cfg=configuration(yaml.safe_load(Path(args.config).read_text()))
    try:
        if args.command=='demo-manifest':save_json(args.out,demo_manifest(args.demo_dir));return
        if args.command=='validate':
            m=validate(args.manifest,args.out);emit(f'Validated {len(m["tiles"])} tiles');return
        if args.command in ('register','run'):
            register(args.manifest,args.out,cfg,emit)
            if args.command=='register':return
            result=render(args.out,Path(args.out)/'output',args.refiner or (cfg or configuration())['refinement']['backend'],cfg,emit)
        elif args.command in ('render','refine'):
            if not args.run:raise ValueError('--run must reference a completed registration directory')
            result=render(args.run,args.out,args.refiner or (cfg or configuration())['refinement']['backend'],cfg,emit)
        elif args.dataset:
            import yaml
            dataset=json.loads(Path(args.dataset).read_text());base=Path(args.dataset).resolve().parent
            matrix=yaml.safe_load(Path(args.matrix).read_text()) if args.matrix else {'baseline':configuration()}
            paired=[]
            for slide_index,slide in enumerate(dataset['slides']):
                for name,settings in matrix.items():
                    if Path(name).name!=name:raise ValueError('Matrix names must be simple directory names')
                    destination=Path(args.out)/f'slide_{slide_index:03d}'/name
                    manifest_path=base/slide['manifest']
                    try:
                        register(manifest_path,destination/'coarse',settings,emit)
                        summary=render(destination/'coarse',destination/'output',configuration(settings)['refinement']['backend'],settings,emit)
                        paired.append(dict(slide=slide_index,split=slide.get('split','unspecified'),pipeline=name,status=summary['status'],landmarks=summary['landmark_metrics'],refinement_outcomes=summary['refinement_outcomes']))
                    except Exception as exc:paired.append(dict(slide=slide_index,pipeline=name,status='failed',reason=str(exc)))
            result={'track':'complete_pipeline','status':'complete','qc_accepted':False,'results':paired}
            save_json(Path(args.out)/'benchmark.json',result)
        else:
            coarse=args.run or str(Path(args.out)/'coarse')
            if not args.run:register(args.manifest,coarse,cfg,emit)
            results=[]
            for backend in args.refiners.split(','):
                results.append(render(coarse,Path(args.out)/backend,backend,cfg,emit))
            result={'track':'fixed_coarse_refinement','status':'complete','qc_accepted':False,'results':results,'note':'All attempts retained. No held-out microscopy accuracy claims.'}
            save_json(Path(args.out)/'benchmark.json',result)
        emit(json.dumps({'status':result['status'],'output':args.out}))
    except Exception as exc:
        save_json(Path(args.out)/'error.json',{'status':'failed','reason':str(exc)})
        print(str(exc),file=sys.stderr);sys.exit(1)

if __name__=='__main__':main()
