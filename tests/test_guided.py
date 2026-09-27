import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from server import app
from guided_stitching import GrowingMap


class GuidedTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(62)
        # Structured landmarks exercise the learned detector; white noise has no
        # stable landmarks for testing learned correspondence matching.
        texture=np.full((400,1200,3),225,np.uint8)
        for _ in range(700):
            x,y=rng.integers([0,0],[1200,400])
            radius=int(rng.integers(2,10));color=tuple(int(v) for v in rng.integers(0,220,3))
            cv2.circle(texture,(int(x),int(y)),radius,color,-1)
        for x in range(0,1200,100):
            cv2.putText(texture,str(x),(x,200),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
        self.images=[texture[:,x:x+500].copy() for x in (0,350,700)]
        for name in ('SIFT_create','ORB_create','BFMatcher','findHomography','estimateAffine2D','estimateAffinePartial2D','warpPerspective','warpAffine'):
            blocker=patch('cv2.'+name,side_effect=AssertionError('Classical matching must not run'))
            blocker.start();self.addCleanup(blocker.stop)

    def test_classical_engines_rejected(self):
        with TestClient(app) as client:
            for engine in ('sift','orb','xfeat','superglue'):
                files=[('files',(f'{i}.png',cv2.imencode('.png',im)[1].tobytes(),'image/png')) for i,im in enumerate(self.images[:2])]
                self.assertEqual(client.post('/api/guided',files=files,data={'engine':engine}).status_code,400)
                with self.assertRaises(ValueError):GrowingMap(engine)

    def test_growth_left_and_right(self):
        mosaic=GrowingMap();mosaic.seed(self.images[1],1)
        for index in (0,2):
            proposal=mosaic.propose(self.images[index],index)
            self.assertGreater(proposal['metrics']['added_pixels'],50000)
            mosaic.commit(proposal)
        self.assertAlmostEqual(mosaic.image.shape[1],1200,delta=5)
        self.assertAlmostEqual(mosaic.image.shape[0],400,delta=5)
        self.assertEqual([p['index'] for p in mosaic.placed],[1,0,2])
        self.assertEqual(mosaic.image.shape[:2],(400,1200))
        for placed in mosaic.placed:
            self.assertEqual(placed['offset'].shape,(2,))
            self.assertTrue(np.issubdtype(placed['offset'].dtype,np.integer))
        # Nonoverlapping content is copied exactly, without warping/resampling.
        np.testing.assert_array_equal(mosaic.image[:,:300],self.images[0][:,:300])

    def test_vertical_growth(self):
        mosaic=GrowingMap()
        mosaic.seed(np.transpose(self.images[1],(1,0,2)).copy(),1)
        for i in (0,2):
            mosaic.commit(mosaic.propose(np.transpose(self.images[i],(1,0,2)).copy(),i))
        self.assertEqual(mosaic.image.shape[:2],(1200,400))

    def test_different_resolutions_and_blank_image(self):
        mosaic=GrowingMap()
        a=mosaic.features(self.images[0])
        enlarged=cv2.resize(self.images[1],(1000,800))
        b=mosaic.features(enlarged)
        x,y,confidence=mosaic.engine.match(a,b)
        self.assertGreater(len(x),12)
        errors=np.linalg.norm(x-(y/2+np.array([350,0])),axis=1)
        self.assertLess(np.median(errors),3)
        self.assertTrue(np.isfinite(confidence).all())
        mosaic.seed(self.images[0],0)
        before=mosaic.image.copy()
        with self.assertRaisesRegex(ValueError,'No reliable translation-only overlap'):
            mosaic.propose(np.full_like(self.images[1],128),1)
        np.testing.assert_array_equal(mosaic.image,before)

    def test_edge_filter_matches_without_changing_original_colors(self):
        mosaic=GrowingMap(edge_filter=True)
        mosaic.seed(self.images[0],0)
        edges=mosaic.placed[0]['features'].small_gray
        self.assertEqual(set(np.unique(edges)),{0,255})
        np.testing.assert_array_equal(mosaic.image,self.images[0])
        proposal=mosaic.propose(self.images[1],1)
        self.assertGreater(proposal['metrics']['inliers'],12)
        mosaic.commit(proposal)
        self.assertAlmostEqual(mosaic.image.shape[1],850,delta=5)
        self.assertTrue(np.any(mosaic.image[:,:,0]!=mosaic.image[:,:,1]))

    def test_edge_filter_session_persistence(self):
        with tempfile.TemporaryDirectory() as temp,patch('guided_stitching.STORE',Path(temp)),TestClient(app) as client:
            files=[('files',(f'{i}.png',cv2.imencode('.png',im)[1].tobytes(),'image/png')) for i,im in enumerate(self.images[:2])]
            state=client.post('/api/guided',files=files,data={'edge_filter':'true'}).json()
            self.assertIs(state['edge_filter'],True)
            url=f"/api/guided/{state['id']}"
            seed=client.post(url+'/select/0').json()
            self.assertTrue(seed['ok'],seed['message'])
            result=client.post(url+'/select/1').json()
            self.assertTrue(result['ok'],result['message'])
            self.assertIs(client.get(url).json()['edge_filter'],True)
            self.assertIs(client.post(url+'/reset').json()['edge_filter'],True)

    def test_no_extension_is_not_committed(self):
        mosaic=GrowingMap();mosaic.seed(self.images[0],0)
        before=mosaic.image.copy()
        with self.assertRaisesRegex(ValueError,'no meaningful new coverage'):mosaic.propose(self.images[0],1)
        np.testing.assert_array_equal(mosaic.image,before)
        self.assertEqual(len(mosaic.placed),1)

    def test_upload_seed_failure_retry_completion_and_reset(self):
        with tempfile.TemporaryDirectory() as temp,patch('guided_stitching.STORE',Path(temp)),TestClient(app) as client:
            files=[('files',(f'{i}.png',cv2.imencode('.png',im)[1].tobytes(),'image/png')) for i,im in enumerate(self.images)]
            response=client.post('/api/guided',files=files,data={'engine':'efficientloftr'})
            self.assertEqual(response.status_code,200,response.text)
            state=response.json();id=state['id']
            self.assertEqual(state['used'],[])
            self.assertIs(state['edge_filter'],False)
            def select(index):return client.post(f'/api/guided/{id}/select/{index}')
            state=select(0).json();self.assertTrue(state['ok']);original=client.get(state['mosaic']).content
            failed=select(2).json();self.assertFalse(failed['ok']);self.assertEqual(failed['used'],[0]);self.assertEqual(failed['version'],1)
            self.assertEqual(client.get(failed['mosaic']).content,original)
            self.assertTrue(select(1).json()['ok'])
            state=select(2).json();self.assertTrue(state['ok']);self.assertTrue(state['complete']);self.assertEqual(state['used'],[0,1,2]);self.assertEqual(state['version'],3)
            self.assertEqual(select(2).status_code,409)
            restored=client.get(f'/api/guided/{id}').json();self.assertTrue(restored['complete'])
            reset=client.post(f'/api/guided/{id}/reset').json();self.assertEqual(reset['used'],[]);self.assertIsNone(reset['mosaic']);self.assertEqual(len(reset['items']),3)
            self.assertTrue(select(2).json()['ok'])

if __name__=='__main__':unittest.main()
