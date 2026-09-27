import json
import time
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from server import app

class ComparisonTests(unittest.TestCase):
    def test_transform_chain_order(self):
        from stitcher import PanoramaStitcher
        a = np.array([[1.1, .2, 50], [.1, .9, 10], [.0002, 0, 1]])
        b = np.array([[.8, -.1, 15], [.2, 1.2, 30], [0, .0003, 1]])
        left = PanoramaStitcher._global_homographies([a, b], 2)
        right = PanoramaStitcher._global_homographies([a, b], 0)
        np.testing.assert_allclose(left[0], b @ a)
        np.testing.assert_allclose(right[2], np.linalg.inv(a) @ np.linalg.inv(b))

    def test_failure_does_not_stop_next_engine(self):
        from unittest.mock import patch
        from comparison import run, records, RUNS
        from types import SimpleNamespace
        import uuid
        id = uuid.uuid4().hex
        (RUNS/id).mkdir(parents=True)
        records[id] = {'status':'queued', 'results':[]}
        result = SimpleNamespace(panorama=np.zeros((20,20,3),np.uint8), stats={'output':{'w':20,'h':20}})
        with patch('comparison.PanoramaStitcher') as stitcher:
            stitcher.return_value.stitch_paths.side_effect = [RuntimeError('missing checkpoint'), result]
            run(id, [], ['xfeat','sift'], {})
        self.assertEqual([r['status'] for r in records[id]['results']], ['failed','complete'])

    def test_upload_comparison_and_validation(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertGreater(len(client.get('/api/engines').json()['engines']), 10)
            demos = client.get('/api/demos').json()['datasets']
            self.assertEqual(len(demos[0]['files']), 36)
            self.assertEqual(client.get('/api/demos/no/no').status_code, 404)
            self.assertEqual(client.post('/api/comparisons', data={'options':'{"engines":["bad"]}'}).status_code, 400)
            rng = np.random.default_rng(7)
            canvas = rng.integers(0,256,(280,700,3),dtype=np.uint8)
            canvas = cv2.GaussianBlur(canvas,(3,3),0)
            files = [('files',(f'{i}.png',cv2.imencode('.png',img)[1].tobytes(),'image/png')) for i,img in enumerate([canvas[:,:460],canvas[:,220:]])]
            response=client.post('/api/comparisons',data={'options':json.dumps({'engines':['sift','orb'],'feature_max_dim':512})},files=files)
            self.assertEqual(response.status_code,200,response.text)
            id=response.json()['id']
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                result=client.get('/api/comparisons/'+id).json()
                if result['status']=='complete': break
                time.sleep(.1)
            self.assertEqual(result['status'],'complete')
            self.assertEqual([r['status'] for r in result['results']],['complete','complete'],result)
            for row in result['results']:
                self.assertGreater(sum(row['inliers']),10)
                self.assertEqual(client.get(row['image']).status_code,200)
            self.assertEqual(client.get(f'/api/comparisons/{id}/report').status_code,200)

if __name__=='__main__': unittest.main()
