#!/usr/bin/env python3
"""Reproduce boundary/full-grid matching evaluations without rendering a mosaic.

Acceptance is geometric consistency, not ground-truth correctness. Pair artifacts
support manual assessment of repetitive tissue and false bridges.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import resource
import time
import cv2
from stitcher.catalog import CATALOG
from stitcher.contract import prepare_image
from stitcher.diagnostics import evaluate_pair
from stitcher.grid import require_positions, DEFAULT_TEMPLATE
from stitcher.matching import build_engine
from stitcher.stitcher import StitchConfig
from stitcher.verification import smoke_record
import numpy as np


def evaluate(dataset,engine,output,full=False,resolution=512):
    paths=sorted(dataset.glob('tile_*.png'))
    positions=require_positions([p.name for p in paths],DEFAULT_TEMPLATE)
    neighbors=[(i,j) for i,j in combinations(range(len(paths)),2)
               if abs(positions[i][0]-positions[j][0])+abs(positions[i][1]-positions[j][1])==1]
    boundaries=[(i,j) for i,j in neighbors if {positions[i][0],positions[j][0]} in ({3,4},{5,6})]
    controls=[p for p in neighbors if p not in boundaries][:6]
    negatives=[(i,j) for i,j in combinations(range(len(paths)),2)
               if positions[i][0]==positions[j][0] and abs(positions[i][1]-positions[j][1])>=10][:3]
    pairs=list(dict.fromkeys((neighbors if full else boundaries+controls)+negatives))
    output.mkdir(parents=True,exist_ok=True)
    cfg=StitchConfig(engine=engine,engine_options={'version':1,'device':'cpu','feature_max_dim':resolution},
        alignment='translation',grid_positions=positions,input_names=[p.name for p in paths],
        weights_dir=str(Path(__file__).resolve().parent/'weights'),diagnostics_dir=str(output/'pairs'),viz=True)
    started=time.perf_counter()
    model=build_engine(cfg.to_dict())
    features={i:prepare_image(model,cv2.imread(str(paths[i])),paths[i].name) for i in sorted({i for p in pairs for i in p})}
    preparation=time.perf_counter()-started
    results=[]
    for i,j in pairs:
        try:
            _,row,_=evaluate_pair(model,features[i],features[j],cfg,[i+1,j+1])
        except Exception as exc:
            row=dict(pair=[i+1,j+1],accepted=False,reason=str(exc))
        row['role']='expected_nonoverlap' if (i,j) in negatives else 'boundary' if (i,j) in boundaries else 'neighbor_control'
        results.append(row)
        (output/'pairs-progress.json').write_text(json.dumps(results,indent=2))
    # Connectivity and edge consistency are assessed only from intended neighbors.
    adjacency={i:[] for i in range(len(paths))}
    for pair,row in zip(pairs,results):
        if pair not in neighbors or not row['accepted']: continue
        i,j=pair; shift=np.asarray(row['transform'])[:2,2]
        adjacency[i].append((j,shift));adjacency[j].append((i,-shift))
    offsets={};groups=[];cycle_errors=[]
    for start in adjacency:
        if start in offsets: continue
        offsets[start]=np.zeros(2);queue=[start]
        for i in queue:
            for j,shift in adjacency[i]:
                proposal=offsets[i]+shift
                if j not in offsets:offsets[j]=proposal;queue.append(j)
                else:cycle_errors.append(float(np.linalg.norm(offsets[j]-proposal)))
        groups.append([i+1 for i in queue])
    report=dict(engine=engine,options=cfg.engine_options,dataset=str(dataset),inputs=[p.name for p in paths],
        positions=positions,full_grid=full,intended_neighbor_pairs=len(neighbors),tested_pairs=len(pairs),
        boundary_pairs=len(boundaries),boundary_accepted=sum(r['accepted'] for r in results if r['role']=='boundary'),
        negative_pairs=len(negatives),negative_accepted=sum(r['accepted'] for r in results if r['role']=='expected_nonoverlap'),
        accepted_neighbors=sum(r['accepted'] for p,r in zip(pairs,results) if p in neighbors),
        components=groups,cycle_residual_max_px=max(cycle_errors,default=0),
        elapsed=time.perf_counter()-started,preparation_seconds=preparation,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        verification=smoke_record(engine),pairs=results,
        limitation='No manual landmark ground truth. Accepted edges are geometric consistency results, not proven correct recovery. Far-apart grid tiles are expected negatives.')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','inputs','positions','verification')},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--engines',nargs='+',default=['sift','xfeat','xfeat-lighterglue','aliked-lightglue','efficientloftr'],choices=list(CATALOG))
    p.add_argument('--full',action='store_true')
    p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--output',type=Path,default=Path('uploads/classic-evaluation'))
    args=p.parse_args()
    for engine in args.engines:
        evaluate(args.dataset,engine,args.output/engine,args.full,args.resolution)
