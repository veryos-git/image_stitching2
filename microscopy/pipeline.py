from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import html
import json
import platform
import shutil
import time
import cv2
import numpy as np
from . import __version__
from .io import validate,save_json,load_image,preview,neighbors,digest
from .contracts import CoarseRequest,RefineRequest,WarpSolveRequest
from .registration import COARSE,assess,overlap
from .optimizer import solve
from .refiners import REFINERS,validate_result
from .mesh import MeshReconciler,displacement
from .render import render_component

DEFAULTS={'schema_version':'1.0','coarse':{'primary':'phase_correlation','fallback':'feature_translation'},'registration_max_dim':1024,'recovery':{'widened_search_attempts':1,'alternative_backend_attempts':1,'graph_retry_passes':1},'refinement':{'backend':'identity','timeout_seconds':90,'acceptance_profile':None},'render':{'chunk_size':512},'optimization':{'outlier_px':8.}}


def configuration(user=None):
    cfg=json.loads(json.dumps(DEFAULTS))
    for key,value in (user or {}).items():
        if key not in cfg:raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(value,dict):cfg[key].update(value)
        else:cfg[key]=value
    if cfg['schema_version']!='1.0':raise ValueError('Unsupported configuration schema')
    if cfg['coarse']['primary'] not in COARSE or cfg['coarse']['fallback'] not in COARSE:raise ValueError('Unknown coarse backend')
    if cfg['refinement']['backend'] not in REFINERS:raise ValueError('Unknown refinement backend')
    if not 128<=cfg['registration_max_dim']<=1600:raise ValueError('registration_max_dim must be 128–1600')
    if not 64<=cfg['render']['chunk_size']<=2048:raise ValueError('chunk_size must be 64–2048')
    for k in ('widened_search_attempts','alternative_backend_attempts','graph_retry_passes'):
        if cfg['recovery'][k] not in (0,1):raise ValueError('Recovery budgets must be 0 or 1')
    if not 1<=float(cfg['refinement']['timeout_seconds'])<=300:raise ValueError('refinement timeout must be 1–300 seconds')
    profile=cfg['refinement'].get('acceptance_profile')
    if profile:
        for k in ('max_displacement_px','max_strain','max_constraint_error_px'):
            if not np.isfinite(float(profile[k])) or float(profile[k])<=0:raise ValueError('Acceptance limits must be positive finite values')
    return cfg


def register(manifest_path,out,config=None,emit=lambda s:None):
    start=time.perf_counter();cfg=configuration(config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=validate(manifest_path);tiles=manifest['tiles']
    key=hashlib.sha256(json.dumps({'manifest':manifest,'coarse':cfg['coarse'],'recovery':cfg['recovery'],'preview':cfg['registration_max_dim'],'optimization':cfg['optimization'],'version':__version__,'opencv':cv2.__version__,'implementation_hash':digest(__file__)+digest(Path(__file__).with_name('registration.py'))},sort_keys=True).encode()).hexdigest()
    if (out/'registration.json').exists():
        cached=json.loads((out/'registration.json').read_text())
        if cached.get('cache_key')==key and cached.get('stage_complete'):
            emit('Reusing completed registration');return cached
    save_json(out/'run.json',dict(schema_version='1.0',status='running',stage='registration',config=cfg))
    images=[];scales=[];masks=[]
    for t in tiles:
        image=load_image(t['path']);small,meta=preview(image,cfg['registration_max_dim']);t['preprocessing']=meta
        mask=None
        if t.get('mask'):
            mask=cv2.resize(load_image(t['mask']),small.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            small=small.copy();small[~mask]=0
        masks.append(mask)
        images.append(small);scales.append(np.array([meta['original_to_preview'][0][0],meta['original_to_preview'][1][1]]))
    save_json(out/'manifest.json',manifest)
    attempts=[];edges=[]
    def attempt(i,j,backend,prior,stage,radius=.25):
        begin=time.perf_counter();direction=(tiles[j]['row']-tiles[i]['row'],tiles[j]['column']-tiles[i]['column'])
        # Resample B into A's preview pixel scale so translation units agree.
        a=images[i];b=cv2.resize(images[j],(round(tiles[j]['width']*scales[i][0]),round(tiles[j]['height']*scales[i][1])),interpolation=cv2.INTER_AREA)
        request=CoarseRequest(a,b,tiles[i],tiles[j],direction,None if prior is None else np.array(prior)*scales[i],radius)
        request.mask_a=masks[i]
        request.mask_b=None if masks[j] is None else cv2.resize(masks[j].astype(np.uint8),b.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
        result=COARSE[backend]().register(request)
        candidates=[]
        for c in result.candidates:
            c=dict(c);c['displacement']=(np.array(c['displacement'])/scales[i]).tolist()
            if c['uncertainty_px'] is not None:c['uncertainty_px']/=float(scales[i].mean())
            candidates.append(c)
        if result.status=='accepted':
            original_a=preview(load_image(tiles[i]['path']),max(tiles[i]['width'],tiles[i]['height']))[0]
            original_b=preview(load_image(tiles[j]['path']),max(tiles[j]['width'],tiles[j]['height']))[0]
            check_request=CoarseRequest(original_a,original_b,tiles[i],tiles[j],direction)
            check_request.mask_a=load_image(tiles[i]['mask'])>0 if tiles[i].get('mask') else None
            check_request.mask_b=load_image(tiles[j]['mask'])>0 if tiles[j].get('mask') else None
            check=assess(original_a,original_b,candidates[0]['displacement'],check_request)
            candidates[0]['original_resolution_check']=check
            if check['ncc']<.45 or check['texture']<4 or check['overlap_pixels']<64:
                result.status='failed';result.reason='original_resolution_validation_failed'
        record=dict(schema_version='1.0',i=i,j=j,tile_ids=[tiles[i]['id'],tiles[j]['id']],backend=backend,backend_version=COARSE[backend].capabilities,attempt=stage,status=result.status,reason=result.reason,candidates=candidates,elapsed=time.perf_counter()-begin,prior=prior,prior_is_measurement=False,search_radius_fraction=radius,preprocessing=[tiles[i]['preprocessing'],tiles[j]['preprocessing']],direction='P_j-P_i')
        if result.status=='accepted':
            crops=overlap(original_a,original_b,candidates[0]['displacement'])
            if crops:
                ca,cb,_=crops
                debug=np.concatenate([ca,cb,cv2.absdiff(ca,cb)],axis=1)
                debug=cv2.resize(debug,(min(1500,debug.shape[1]),max(1,round(debug.shape[0]*min(1.,1500/debug.shape[1])))))
                (out/'debug').mkdir(exist_ok=True);debug_name=f'debug/edge_{i}_{j}_{len(attempts)}.jpg'
                cv2.imwrite(str(out/debug_name),debug);record['debug_preview']=debug_name
        attempts.append(record)
        # Preserve ambiguity; topology alone never turns prediction into measurement.
        if result.status=='accepted':
            c=candidates[0];record['accepted_displacement']=c['displacement']
            return dict(i=i,j=j,status='accepted',displacement=c['displacement'],backend=backend,uncertainty_px=c['uncertainty_px'],evidence='image_supported',attempt_index=len(attempts)-1)
        return None
    pairs=neighbors(tiles)
    for k,(i,j) in enumerate(pairs):
        emit(f'Registering grid neighbor {k+1}/{len(pairs)}')
        grid=manifest.get('grid',{});prior=None
        ox,oy=grid.get('nominal_overlap_fraction_x'),grid.get('nominal_overlap_fraction_y')
        if tiles[j]['column']!=tiles[i]['column'] and ox is not None:prior=[tiles[i]['width']*(1-float(ox)),0]
        elif oy is not None:prior=[0,tiles[i]['height']*(1-float(oy))]
        edge=attempt(i,j,cfg['coarse']['primary'],prior,'primary')
        if edge is None and prior is not None and cfg['recovery']['widened_search_attempts']:edge=attempt(i,j,cfg['coarse']['primary'],None,'widened_search',1.)
        ambiguous=any(a['status']=='ambiguous' for a in attempts if a['i']==i and a['j']==j)
        if edge is None and not ambiguous and cfg['recovery']['alternative_backend_attempts']:edge=attempt(i,j,cfg['coarse']['fallback'],None,'alternative_backend',1.)
        edges.append(edge or dict(i=i,j=j,status='ambiguous' if ambiguous else 'failed',evidence='unresolved'))
    poses=solve(len(tiles),edges,cfg['optimization']['outlier_px'])
    if cfg['recovery']['graph_retry_passes']:
        group={i:g for g,ids in enumerate(poses['components']) for i in ids}
        for k,e in enumerate(edges):
            i,j=e['i'],e['j']
            if e['status']!='accepted' and group[i]==group[j]:
                prior=(np.array(poses['positions'][j])-poses['positions'][i]).tolist()
                recovered=attempt(i,j,cfg['coarse']['primary'],prior,'graph_informed_retry',.025)
                if recovered:edges[k]=recovered
        poses=solve(len(tiles),edges,cfg['optimization']['outlier_px'])
    if cfg['recovery']['graph_retry_passes'] and len(poses['components'])>1:
        horizontal=[e['displacement'] for e in edges if e['status']=='accepted' and tiles[e['j']]['row']==tiles[e['i']]['row'] and tiles[e['j']]['column']-tiles[e['i']]['column']==1]
        vertical=[e['displacement'] for e in edges if e['status']=='accepted' and tiles[e['j']]['column']==tiles[e['i']]['column'] and tiles[e['j']]['row']-tiles[e['i']]['row']==1]
        if len(horizontal)>=2 and len(vertical)>=2:
            stride_x=np.median(horizontal,axis=0);stride_y=np.median(vertical,axis=0)
            known={tuple(sorted((e['i'],e['j']))) for e in edges}
            groups={i:k for k,g in enumerate(poses['components']) for i in g}
            for i in range(len(tiles)):
                for j in range(i+1,len(tiles)):
                    if (i,j) in known or groups[i]==groups[j]:continue
                    dr=tiles[j]['row']-tiles[i]['row'];dc=tiles[j]['column']-tiles[i]['column']
                    predicted=dr*stride_y+dc*stride_x
                    overlap_area=max(0,tiles[i]['width']-abs(predicted[0]))*max(0,tiles[i]['height']-abs(predicted[1]))
                    if overlap_area<.03*tiles[i]['width']*tiles[i]['height']:continue
                    recovered=attempt(i,j,cfg['coarse']['primary'],predicted.tolist(),'physically_plausible_recovery',.05)
                    edges.append(recovered or dict(i=i,j=j,status='failed',evidence='unresolved'))
            poses=solve(len(tiles),edges,cfg['optimization']['outlier_px'])
    poses['tiles']=[dict(id=t['id'],component=next(g for g,c in enumerate(poses['components']) if i in c),evidence='image_supported' if any(e['status']=='accepted' and i in (e['i'],e['j']) for e in edges) else 'unresolved',position=poses['positions'][i]) for i,t in enumerate(tiles)]
    save_json(out/'poses.json',poses)
    with open(out/'edges.jsonl','w') as f:
        for a in attempts:f.write(json.dumps(a)+'\n')
    result=dict(schema_version='1.0',stage_complete=True,cache_key=key,manifest=manifest,config=cfg,edges=edges,poses=poses,elapsed=time.perf_counter()-start)
    save_json(out/'registration.json',result)
    save_json(out/'run.json',dict(schema_version='1.0',status='registered',config=cfg,registration_seconds=result['elapsed']))
    return result


def render(run_dir,out=None,refiner='identity',config=None,emit=lambda s:None):
    start=time.perf_counter();run_dir=Path(run_dir);reg=json.loads((run_dir/'registration.json').read_text())
    if not reg.get('stage_complete'):raise ValueError('Registration stage is incomplete')
    cfg=configuration(config or reg['config']);cfg['refinement']['backend']=refiner
    if refiner not in REFINERS:raise ValueError('Unknown refiner')
    for tile in reg['manifest']['tiles']:
        if digest(tile['path'])!=tile['sha256']:raise ValueError('Original changed since registration: '+tile['id'])
        if tile.get('mask') and digest(tile['mask'])!=tile['mask_sha256']:raise ValueError('Mask changed since registration: '+tile['id'])
    out=Path(out or run_dir/refiner);out.mkdir(parents=True,exist_ok=True)
    tiles=reg['manifest']['tiles'];poses=reg['poses'];backend=REFINERS[refiner]();constraints=[];refinements=[]
    save_json(out/'run.json',dict(status='running',stage='refinement',config=cfg))
    deployed_model=None
    if refiner=='pixelstitch':
        checkpoint=Path(__file__).resolve().parents[1]/'vendor/PixelStitch/checkpoints/ckpt.pth'
        deployed_model=digest(checkpoint) if checkpoint.exists() else 'missing'
    cache=run_dir/'refinement_cache';cache.mkdir(exist_ok=True)
    for e in reg['edges']:
        if e['status']!='accepted':continue
        i,j=e['i'],e['j'];emit(f'{refiner}: overlap {tiles[i]["id"]} ↔ {tiles[j]["id"]}')
        identity=dict(deployed_model=deployed_model,tiles=[tiles[i]['sha256'],tiles[j]['sha256']],masks=[tiles[i].get('mask_sha256'),tiles[j].get('mask_sha256')],positions=[poses['positions'][i],poses['positions'][j]],implementation_hash=digest(__file__)+digest(Path(__file__).with_name('refiners.py'))+digest(Path(__file__).with_name('pixel_worker.py')),config=cfg['refinement'],backend=backend.capabilities,version=__version__)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();cached=cache/(key+'.json')
        if cached.exists():record=json.loads(cached.read_text());record['cache_hit']=True
        else:
            tick=time.perf_counter()
            try:
                # Refiners work on full-resolution normalized grayscale, not output intensities.
                aa=preview(load_image(tiles[i]['path']),max(tiles[i]['width'],tiles[i]['height']))[0]
                bb=preview(load_image(tiles[j]['path']),max(tiles[j]['width'],tiles[j]['height']))[0]
                d=np.array(poses['positions'][j])-poses['positions'][i]
                r=validate_result(backend.refine(RefineRequest(aa,bb,d,cfg['refinement'])))
                record=dict(i=i,j=j,backend=refiner,status=r.status,diagnostics=r.diagnostics,frame=r.frame,a=None if r.points_a is None else np.asarray(r.points_a).tolist(),b=None if r.points_b is None else np.asarray(r.points_b).tolist(),elapsed=time.perf_counter()-tick,cache_hit=False)
                if r.points_a is not None:
                    # Independent photometric gate (not held-out landmark accuracy).
                    pa,pb=np.array(record['a']),np.array(record['b']);valid=np.ones(len(pa),bool)
                    for tile,points in ((tiles[i],pa),(tiles[j],pb)):
                        valid&=(points[:,0]>=1)&(points[:,0]<tile['width']-2)&(points[:,1]>=1)&(points[:,1]<tile['height']-2)
                        if tile.get('mask'):
                            mask=load_image(tile['mask']);valid&=cv2.remap(mask.astype(np.float32),points[:,0].astype(np.float32)[:,None],points[:,1].astype(np.float32)[:,None],cv2.INTER_NEAREST).ravel()>0
                    pa,pb=pa[valid],pb[valid]
                    def sample(im,p):return cv2.remap(im.astype(np.float32),p[:,0].astype(np.float32)[:,None],p[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR).ravel()
                    baseline=np.mean(abs(sample(aa,pa)-sample(bb,pa-d))) if len(pa) else float('inf')
                    refined=np.mean(abs(sample(aa,pa)-sample(bb,pb))) if len(pa) else float('inf')
                    if len(pa)<12 or not refined<baseline*.98:
                        record.update(status='no_improvement',a=None,b=None);record['diagnostics']['rejection']='no_independent_photometric_improvement'
                    else:record.update(a=pa.tolist(),b=pb.tolist());record['diagnostics'].update(baseline_mae=float(baseline),refined_mae=float(refined))
            except Exception as exc:record=dict(i=i,j=j,backend=refiner,status='failed',diagnostics={'reason':str(exc)},a=None,b=None,elapsed=time.perf_counter()-tick)
            save_json(cached,record)
        refinements.append(record)
        if record['status']=='success' and record['a'] is not None:constraints.append(record)
    warps=MeshReconciler().solve(WarpSolveRequest(tiles,poses['positions'],constraints,cfg['refinement']))
    save_json(out/'warps/maps.json',dict(convention=warps.convention,meshes=[m.tolist() for m in warps.meshes],validation=warps.validation))
    save_json(out/'refinements.json',refinements)
    components=[]
    for index,ids in enumerate(poses['components']):
        emit(f'Rendering component {index+1}/{len(poses["components"])}')
        component_out=out/f'component_{index:03d}'
        metadata=render_component([tiles[i] for i in ids],[poses['positions'][i] for i in ids],[warps.meshes[i] for i in ids],component_out,cfg['render']['chunk_size'],emit)
        components.append(dict(id=index,tiles=[tiles[i]['id'] for i in ids],placement='image_supported' if len(ids)>1 else 'unresolved_singleton',path=component_out.name,**metadata))
    landmark_errors=[]
    index_by_id={t['id']:i for i,t in enumerate(tiles)}
    for landmark in reg['manifest'].get('evaluation_landmarks',[]):
        i,j=index_by_id[landmark['tile_a']],index_by_id[landmark['tile_b']]
        if poses['tiles'][i]['component']!=poses['tiles'][j]['component']:continue
        a,b=np.asarray(landmark['point_a'],float),np.asarray(landmark['point_b'],float)
        fa=np.asarray(poses['positions'][i])+a+displacement(warps.meshes[i],a,tiles[i]['width'],tiles[i]['height'])
        fb=np.asarray(poses['positions'][j])+b+displacement(warps.meshes[j],b,tiles[j]['width'],tiles[j]['height'])
        landmark_errors.append(float(np.linalg.norm(fa-fb)))
    landmark_metrics={'count':len(landmark_errors),'median_px':float(np.median(landmark_errors)) if landmark_errors else None,'p95_px':float(np.percentile(landmark_errors,95)) if landmark_errors else None,'use':'evaluation only; never registration or acceptance'}
    unresolved=[t['id'] for t in poses['tiles'] if t['evidence']=='unresolved']
    status='complete' if len(components)==1 and not unresolved else 'partial'
    if not any(len(c['tiles'])>1 for c in components):status='failed'
    summary=dict(schema_version='1.0',status=status,qc_accepted=False,qc_reason='No locked-set scientific acceptance profile/annotations; engineering diagnostics only',geometry_consistent=poses['qc_consistent'],components=components,review_queue=unresolved+[f'Unresolved relative placement of component {c["id"]}' for c in components[1:]],refiner=refiner,refinement_outcomes={s:sum(r['status']==s for r in refinements) for s in ('success','no_improvement','unsupported_input','failed')},refinement_applied=bool(constraints and warps.validation['accepted']),warp_validation=warps.validation,elapsed=time.perf_counter()-start,registration_seconds=reg['elapsed'],config=cfg,versions=dict(core=__version__,opencv=cv2.__version__,numpy=np.__version__),hardware=dict(platform=platform.platform(),processor=platform.processor()),coarse_run=str(run_dir.resolve()),landmark_metrics=landmark_metrics)
    save_json(out/'summary.json',summary);save_json(out/'run.json',summary)
    save_json(out/'manifest.json',reg['manifest']);save_json(out/'poses.json',poses)
    if (run_dir/'debug').exists():shutil.copytree(run_dir/'debug',out/'debug',dirs_exist_ok=True)
    (out/'edges.jsonl').write_text((run_dir/'edges.jsonl').read_text())
    with open(out/'metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=['i','j','residual_px','flagged']);writer.writeheader();writer.writerows(poses['residuals'])
    previews=''.join(f'<figure><img style="max-width:100%" src="debug/{p.name}"><figcaption>{p.name}: tile A, aligned tile B, absolute difference</figcaption></figure>' for p in sorted((out/'debug').glob('*.jpg')))
    rows=''.join(f'<tr><td>{html.escape(t["id"])}</td><td>{t["component"]}</td><td>{html.escape(t["evidence"])}</td><td>{html.escape(str(t["position"]))}</td></tr>' for t in poses['tiles'])
    (out/'qc.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Microscope QC</title><h1>{status} · {html.escape(refiner)}</h1><p>QC not accepted: scientific validation remains pending. Components have independent origins.</p><table><tr><th>Tile</th><th>Component</th><th>Evidence</th><th>Position xy</th></tr>{rows}</table><h2>Diagnostics</h2><pre>{html.escape(json.dumps(summary,indent=2))}</pre><h2>Original-resolution overlap checks</h2>{previews}<h2>Registration attempts</h2><pre>{html.escape((run_dir/"edges.jsonl").read_text())}</pre>')
    return summary
