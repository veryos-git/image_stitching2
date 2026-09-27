import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from .contracts import TileWarpSet


def basis(points,w,h,size=5):
    xy=np.clip(np.asarray(points)/[max(w-1,1),max(h-1,1)]*(size-1),0,size-1-1e-7)
    ix=xy[:,0].astype(int);iy=xy[:,1].astype(int);fx=xy[:,0]-ix;fy=xy[:,1]-iy
    indices=np.stack([iy*size+ix,iy*size+ix+1,(iy+1)*size+ix,(iy+1)*size+ix+1],1)
    weights=np.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],1)
    return indices,weights


def displacement(mesh,points,w,h):
    ix,wt=basis(np.asarray(points).reshape(-1,2),w,h,mesh.shape[0])
    return (mesh.reshape(-1,2)[ix]*wt[:,:,None]).sum(1).reshape(np.asarray(points).shape)


class MeshReconciler:
    def solve(self,req):
        n=len(req.tiles);size=5;nodes=size*size
        zero=[np.zeros((size,size,2)) for _ in req.tiles]
        if not req.constraints:return TileWarpSet(zero,{'accepted':True,'reason':'identity'})
        profile=req.config.get('acceptance_profile')
        if not profile:return TileWarpSet(zero,{'accepted':False,'reason':'no_scientific_acceptance_profile; coarse_baseline_retained'})
        max_motion=float(profile['max_displacement_px']);max_strain=float(profile['max_strain']);max_error=float(profile['max_constraint_error_px'])
        rows=[];targets=[]
        for c in req.constraints:
            i,j=c['i'],c['j'];a,b=np.asarray(c['a']),np.asarray(c['b'])
            ia,wa=basis(a,req.tiles[i]['width'],req.tiles[i]['height']);ib,wb=basis(b,req.tiles[j]['width'],req.tiles[j]['height'])
            residual=np.asarray(req.positions[j])+b-np.asarray(req.positions[i])-a
            for k in range(len(a)):
                row={int(i*nodes+v):float(w) for v,w in zip(ia[k],wa[k])}
                for v,w in zip(ib[k],wb[k]):row[int(j*nodes+v)]=row.get(int(j*nodes+v),0)-float(w)
                rows.append(row);targets.append(residual[k])
        count=len(rows)
        for i in range(n):
            # Magnitude/unsupported regions shrink toward zero; constant mode fixed.
            for k in range(nodes):rows.append({i*nodes+k:.2});targets.append([0,0])
            rows.append({i*nodes+k:10/nodes for k in range(nodes)});targets.append([0,0])
            for y in range(size):
                for x in range(size):
                    for yy,xx in ((y+1,x),(y,x+1)):
                        if yy<size and xx<size:
                            rows.append({i*nodes+y*size+x:1.,i*nodes+yy*size+xx:-1.});targets.append([0,0])
        mat=lil_matrix((len(rows),n*nodes))
        for k,row in enumerate(rows):
            for j,v in row.items():mat[k,j]=v
        mat=mat.tocsr();target=np.array(targets)
        weights=np.ones(len(rows))
        for _ in range(3):
            weighted=mat.multiply(weights[:,None]).tocsr()
            solution=np.stack([lsqr(weighted,target[:,axis]*weights,atol=1e-8,btol=1e-8)[0] for axis in range(2)],1)
            errors=np.linalg.norm(mat@solution-target,axis=1)
            weights[:count]=np.minimum(1.,2/np.maximum(errors[:count],1e-8))**.5
        centered=solution.reshape(n,size,size,2)
        centered-=centered.mean(axis=(1,2),keepdims=True)
        solution=centered.reshape(-1,2)
        errors=np.linalg.norm(mat@solution-target,axis=1)
        meshes=list(centered);reason=None;motion=0.;strain=0.
        for tile,m in zip(req.tiles,meshes):
            motion=max(motion,float(np.linalg.norm(m,axis=-1).max()))
            dx=np.diff(m,axis=1)/(max(tile['width']-1,1)/(size-1));dy=np.diff(m,axis=0)/(max(tile['height']-1,1)/(size-1))
            strain=max(strain,float(abs(dx).max()),float(abs(dy).max()))
        error=float(np.max(errors[:count]))
        # Derivative bound <.25 per entry guarantees positive determinant and contraction inversion.
        if not np.isfinite(solution).all():reason='nonfinite_mesh'
        elif motion>max_motion:reason='excessive_displacement'
        elif strain>min(max_strain,.24):reason='excessive_strain_or_fold_risk'
        elif error>max_error:reason='multi_neighbor_constraint_conflict'
        elif np.mean(errors[:count])>=np.mean(np.linalg.norm(target[:count],axis=1))*.98:reason='no_reconciled_improvement'
        report=dict(accepted=reason is None,reason=reason or 'passed_configured_experimental_limits',max_displacement_px=motion,max_strain=strain,max_constraint_error_px=error,scientifically_validated=False)
        return TileWarpSet(zero if reason else meshes,report)
