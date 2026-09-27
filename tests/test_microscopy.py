import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import tifffile
from microscopy.io import neighbors,validate,save_json,digest
from microscopy.optimizer import solve
from microscopy.registration import PhaseCorrelation
from microscopy.contracts import CoarseRequest,RefineRequest,RefineResult,WarpSolveRequest
from microscopy.refiners import Identity,validate_result,corner_offsets
from microscopy.mesh import MeshReconciler,displacement
from microscopy.render import render_component

class MicroscopeTests(unittest.TestCase):
    def test_grid_and_gaps(self):
        tiles=[dict(row=r,column=c) for r in range(4) for c in range(5)]
        pairs=neighbors(tiles);self.assertEqual(len(pairs),31);self.assertEqual(len(set(pairs)),31)
        self.assertEqual(neighbors([dict(row=0,column=0),dict(row=0,column=2)]),[])
    def test_pose_sign_and_recovery(self):
        truth=np.array([[0,0],[80,0],[0,80],[80,80]],float)
        edges=[dict(i=i,j=j,status='accepted',displacement=(truth[j]-truth[i]).tolist()) for i,j in [(0,1),(0,2),(1,3),(2,3)]]
        for subset in (edges,edges[:-1]):np.testing.assert_allclose(solve(4,subset)['positions'],truth,atol=1e-6)
        disconnected=solve(4,edges[:1]);self.assertEqual(len(disconnected['components']),3)
        edges[-1]['displacement']=[110,40];self.assertFalse(solve(4,edges)['qc_consistent'])
    def test_translation_blank_and_periodicity(self):
        rng=np.random.default_rng(17);canvas=cv2.GaussianBlur(rng.integers(0,256,(128,230),dtype=np.uint8),(3,3),0)
        a,b=canvas[:,:150],canvas[:,80:230]
        r=PhaseCorrelation().register(CoarseRequest(a,b,{}, {},(0,1)))
        self.assertEqual(r.status,'accepted');np.testing.assert_allclose(r.candidates[0]['displacement'],[80,0],atol=.5)
        blank=np.zeros_like(a);self.assertEqual(PhaseCorrelation().register(CoarseRequest(blank,blank,{}, {},(0,1))).status,'failed')
        periodic=np.tile(rng.integers(0,256,(128,16),dtype=np.uint8),(1,10))
        r=PhaseCorrelation().register(CoarseRequest(periodic,periodic,{}, {},(0,1)))
        self.assertNotEqual(r.status,'accepted')
    def test_contracts_and_corners(self):
        r=Identity().refine(None);self.assertEqual(r.status,'success');self.assertEqual(r.diagnostics['deformation'],0)
        with self.assertRaises(ValueError):validate_result(RefineResult('success',np.array([[np.nan,0]]),np.zeros((1,2))))
        H=np.eye(3);H[:2,2]=[12,-9];np.testing.assert_allclose(corner_offsets(H,512,256),np.tile([12,-9],(4,1)))
    def test_original_precision_chunk_equivalence_and_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=np.arange(100*140,dtype=np.uint16).reshape(100,140)*3
            a=np.stack([a,a+100,a+200],-1);a[10,10]=0;path=root/'a.tif';tifffile.imwrite(path,a,photometric='rgb')
            mask=np.ones((100,140),np.uint8);mask[0:5,0:5]=0;cv2.imwrite(str(root/'mask.png'),mask)
            tile=dict(id='a',path=str(path),sha256=digest(path),width=140,height=100,dtype='uint16',channels=['a','b','c'],mask=str(root/'mask.png'),mask_sha256=digest(root/'mask.png'))
            mesh=np.zeros((5,5,2));first=render_component([tile],[[0,0]],[mesh],root/'one',512);render_component([tile],[[0,0]],[mesh],root/'chunks',64)
            actual=tifffile.imread(root/'one/mosaic.tif');np.testing.assert_array_equal(actual,tifffile.imread(root/'chunks/mosaic.tif'));np.testing.assert_array_equal(actual[6:],a[6:]);self.assertEqual(actual.dtype,np.uint16)
            coverage=tifffile.imread(root/'one/coverage.tif');self.assertEqual(coverage[1,1],0);self.assertEqual(coverage[10,10],1)
    def test_duplicate_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cv2.imwrite(str(root/'a.png'),np.zeros((30,30),np.uint8))
            manifest=dict(schema_version='1.0',tiles=[dict(id='a',path='a.png',row=0,column=0),dict(id='b',path='a.png',row=0,column=0)])
            save_json(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'unique'):validate(root/'manifest.json')
    def test_resume_and_original_mutation(self):
        from microscopy.pipeline import register,render
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rng=np.random.default_rng(7)
            image=cv2.GaussianBlur(rng.integers(0,256,(120,220),dtype=np.uint8),(3,3),0)
            cv2.imwrite(str(root/'a.png'),image[:,:150]);cv2.imwrite(str(root/'b.png'),image[:,70:])
            manifest=dict(schema_version='1.0',tiles=[dict(id='a',path='a.png',row=0,column=0),dict(id='b',path='b.png',row=0,column=1)])
            save_json(root/'input.json',manifest)
            first=register(root/'input.json',root/'coarse')
            with patch('microscopy.registration.PhaseCorrelation.register',side_effect=AssertionError('cache miss')):
                cached=register(root/'input.json',root/'coarse')
                self.assertEqual(cached['cache_key'],first['cache_key'])
            result=render(root/'coarse',root/'output')
            self.assertEqual(result['status'],'complete');self.assertFalse(result['qc_accepted'])
            cv2.imwrite(str(root/'b.png'),np.zeros((120,150),np.uint8))
            with self.assertRaisesRegex(ValueError,'changed since registration'):render(root/'coarse',root/'again')
    def test_preview_coordinate_roundtrip(self):
        from microscopy.io import preview
        _,meta=preview(np.zeros((123,277),np.uint16),128)
        matrix=np.array(meta['original_to_preview']);points=np.array([[0,0,1],[42.3,87.1,1],[276,122,1]])
        np.testing.assert_allclose((points@matrix.T)@np.linalg.inv(matrix).T,points,atol=1e-12)

    def test_mesh_conflict_falls_back(self):
        tiles=[dict(width=100,height=100)]*2;p=np.array([[20,20],[20,80],[80,20],[80,80]])
        constraints=[dict(i=0,j=1,a=p.tolist(),b=(p+40).tolist()),dict(i=0,j=1,a=p.tolist(),b=(p-40).tolist())]
        config={'acceptance_profile':{'max_displacement_px':2,'max_strain':.1,'max_constraint_error_px':1}}
        result=MeshReconciler().solve(WarpSolveRequest(tiles,[[0,0],[0,0]],constraints,config));self.assertFalse(result.validation['accepted']);self.assertTrue(all(np.all(m==0) for m in result.meshes))

if __name__=='__main__':unittest.main()
