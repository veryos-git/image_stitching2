import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import torch
from stitcher.catalog import CATALOG, engines, resolve_options
from stitcher.contract import PairMatches, prepare_image, match_pair
from stitcher.matching import build_engine
from stitcher.stitcher import StitchConfig, PanoramaStitcher, StitchError
from stitcher.progress import ProgressReporter


class ContractTests(unittest.TestCase):
    def test_invalid_lengths_nonfinite_bounds_and_missing_scores(self):
        with self.assertRaises(ValueError):
            PairMatches(np.zeros((2,2)),np.zeros((1,2)),None).validated((10,10),(10,10))
        with self.assertRaises(ValueError):
            PairMatches(np.zeros((2,2)),np.zeros((2,2)),np.zeros(1)).validated((10,10),(10,10))
        points = np.array([[1.25,2.5],[np.nan,1],[-1,2],[10,2]])
        result = PairMatches(points,points,None).validated((10,10),(10,10))
        self.assertIsNone(result.scores)
        np.testing.assert_array_equal(result.points0,[[1.25,2.5]])
        self.assertEqual(result.metadata['discarded_invalid'],3)
        result = PairMatches(np.empty((0,2)),np.empty((0,2)),None).validated((10,10),(10,10))
        self.assertEqual(result.points0.shape,(0,2))
        self.assertEqual(result.points0.dtype,np.float32)

    def test_rgb_and_pairwise_preview_without_fake_features(self):
        received = []
        def extract(image):
            received.append(image.copy())
            return SimpleNamespace(shape=image.shape[:2],small_gray=np.zeros((32,64),np.uint8))
        model = SimpleNamespace(name='roma',config={'engine':'roma'},extract=extract)
        bgr = np.zeros((103,207,3),np.uint8); bgr[...,0]=17; bgr[...,2]=222
        prepared = prepare_image(model,bgr,'tile')
        np.testing.assert_array_equal(received[0][0,0],[222,0,17])
        np.testing.assert_allclose(prepared.scale,[64/207,32/103])
        self.assertEqual(prepared.keypoints.shape,(0,2))
        self.assertEqual(prepared.shape,(103,207))

    def test_loftr_independent_resize_restoration(self):
        from stitcher.alternatives import AlternativeEngine
        model = AlternativeEngine.__new__(AlternativeEngine)
        model.name='loftr-outdoor'; model.device='cpu'
        model.config={'feature_max_dim':512,'max_keypoints':1024}
        def infer(batch):
            h0,w0=batch['image0'].shape[-2:]; h1,w1=batch['image1'].shape[-2:]
            return dict(keypoints0=torch.tensor([[w0*.5,h0*.25]]),keypoints1=torch.tensor([[w1*.5,h1*.25]]),confidence=torch.tensor([.7]))
        model.model=infer
        a=prepare_image(model,np.zeros((331,577,3),np.uint8))
        b=prepare_image(model,np.zeros((419,293,3),np.uint8))
        out=match_pair(model,a,b)
        np.testing.assert_allclose(out.points0,[[577*.5,331*.25]],rtol=1e-6)
        np.testing.assert_allclose(out.points1,[[293*.5,419*.25]],rtol=1e-6)

    def test_schemas_migration_and_unavailable_components(self):
        self.assertIn('efficientloftr',CATALOG)
        self.assertGreater(len(CATALOG),65)
        migrated=resolve_options('superglue',None,{'superglue_weights':'indoor','sinkhorn_iterations':20})
        self.assertEqual(migrated['superglue_weights'],'indoor')
        self.assertEqual(migrated['version'],1)
        for options in ({'version':2},{'match_threshold':.2},{'max_keypoints':True},{'max_keypoints':3.5},{'max_keypoints':float('nan')},{'unknown':1}):
            with self.assertRaises(ValueError): resolve_options('sift',options)
        for id in ('dad','hardnet','patch2pix','ufm'):
            with self.assertRaises(ValueError): resolve_options(id)
        with patch('stitcher.verification.smoke_record',return_value=None):
            row=next(e for e in engines() if e['id']=='aliked-lightglue')
            self.assertFalse(row['available'])
            self.assertNotEqual(row['status'],'Ready')

    def test_explicit_cuda_never_falls_back(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'CUDA'):
                build_engine({'engine':'superglue','engine_options':{'device':'cuda'}})

    def test_translation_retains_fractional_shift(self):
        from stitcher.unordered import reliable_edge
        x,y=np.meshgrid(np.linspace(20,150,12),np.linspace(20,150,12))
        points=np.column_stack([x.ravel(),y.ravel()])
        H,_,reason=reliable_edge(points,points+[10.25,-3.75],(200,200),(200,200),.5,'translation')
        self.assertIsNone(reason)
        np.testing.assert_allclose(H,[[1,0,10.25],[0,1,-3.75],[0,0,1]])

    def test_matching_failure_preserves_prior_artifacts(self):
        rng=np.random.default_rng(22)
        scene=cv2.GaussianBlur(rng.integers(0,256,(180,500,3),dtype=np.uint8),(3,3),0)
        images=[scene[:,:300],scene[:,100:400],scene[:,200:]]
        events=[]
        from stitcher.contract import match_pair as actual_match
        count=0
        def fail_second(*args):
            nonlocal count
            count+=1
            if count==2: raise RuntimeError('inference failed')
            return actual_match(*args)
        with tempfile.TemporaryDirectory() as directory,patch('stitcher.diagnostics.match_pair',side_effect=fail_second):
            cfg=StitchConfig(engine='sift',pairing='grid',grid_positions=[(0,0),(0,1),(0,2)],disconnected='reject',
                diagnostics_dir=directory,diagnostics_url='/diagnostics')
            with self.assertRaises(StitchError):
                PanoramaStitcher(cfg,ProgressReporter(emit=events.append)).stitch(images)
            self.assertTrue((Path(directory)/'1-2.npz').is_file())
            self.assertTrue((Path(directory)/'2-3.json').is_file())
            pairs=[e['diagnostic'] for e in events if e['type']=='pair_diagnostic']
            self.assertTrue(pairs[0]['accepted'])
            self.assertEqual(pairs[1]['rejection']['code'],'matching_failed')
            with np.load(Path(directory)/'1-2.npz') as data:
                self.assertEqual(len(data['points0']),len(data['inliers']))
                self.assertTrue(data['inliers'].any())

if __name__=='__main__': unittest.main()


class IncrementalLearnedContractTests(unittest.TestCase):
    def test_original_neighbor_payloads_and_rejected_add_leave_map_unchanged(self):
        from stitcher.incremental import IncrementalStitcher
        from stitcher.contract import PairMatches
        model=SimpleNamespace(name='efficientloftr',config={'engine':'efficientloftr'},device='cpu',
            extract=lambda gray:SimpleNamespace(shape=gray.shape,small_gray=gray))
        x,y=np.meshgrid(np.linspace(50,170,14),np.linspace(20,180,14))
        points=np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)
        def matched(engine,new,reference):
            return PairMatches(points,points+[20,0],None)
        cfg=StitchConfig(engine='efficientloftr',alignment='translation',exposure=False,viz=True)
        image=np.random.default_rng(18).integers(0,256,(200,220,3),dtype=np.uint8)
        with patch('stitcher.incremental.build_engine',return_value=model),patch('stitcher.contract.match_pair',side_effect=matched) as match:
            stitcher=IncrementalStitcher(cfg)
            stitcher.seed(image,position=(0,0))
            stitcher.add(image,position=(0,1))
            self.assertEqual(len(stitcher.placed),2)
            np.testing.assert_allclose(stitcher.placed[1]['transform'],[[1,0,20],[0,1,0],[0,0,1]])
            before=stitcher.map_img.copy()
            with self.assertRaisesRegex(StitchError,'neighbor'):
                stitcher.add(image,position=(3,3))
            self.assertEqual(match.call_count,1)
            np.testing.assert_array_equal(before,stitcher.map_img)
            self.assertEqual(stitcher.count,2)
