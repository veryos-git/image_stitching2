import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from server import app
from stitcher.matching import build_engine
from stitcher.catalog import engines


class MatchAnythingTests(unittest.TestCase):
    def test_missing_checkpoint_is_not_advertised_ready(self):
        with patch('stitcher.catalog.ROOT',Path('/nonexistent-matchanything-test')):
            row=next(e for e in engines() if e['id']=='matchanything-eloftr')
            self.assertFalse(row['available'])
            self.assertIn('setup_matchanything.py',row['setup'])

    def test_real_matching_and_comparison(self):
        image=cv2.imread(str(Path(__file__).resolve().parents[1]/'demo_content/scan_2026-09-17_010421/tile_r00_c01.png'))
        image=cv2.resize(image,(960,540))
        images=[image[:,:700],image[:,260:]]
        model=build_engine({'engine':'matchanything-eloftr','feature_max_dim':512})
        a,b=[model.extract(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)) for im in images]
        x,y,confidence=model.match(a,b)
        self.assertGreater(len(x),100)
        self.assertLess(np.median(np.linalg.norm(y-x+[260,0],axis=1)),3)
        self.assertTrue(np.isfinite(confidence).all())
        del model
        with tempfile.TemporaryDirectory() as temp,patch('comparison.RUNS',Path(temp)),TestClient(app) as client:
            row=next(e for e in client.get('/api/engines').json()['engines'] if e['id']=='matchanything-eloftr')
            self.assertTrue(row['available'])
            files=[('files',(f'{i}.png',cv2.imencode('.png',im)[1].tobytes(),'image/png')) for i,im in enumerate(images)]
            r=client.post('/api/comparisons',files=files,data={'options':json.dumps({'engines':['matchanything-eloftr','sift'],'feature_max_dim':512})})
            self.assertEqual(r.status_code,200,r.text)
            id=r.json()['id'];deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                result=client.get('/api/comparisons/'+id).json()
                if result['status']=='complete':break
                time.sleep(.1)
            self.assertEqual(result['status'],'complete',result)
            self.assertEqual([r['status'] for r in result['results']],['complete','complete'],result)
            self.assertGreater(sum(result['results'][0]['inliers']),50)
            self.assertEqual(client.get(result['results'][0]['image']).status_code,200)
            # Completion is published just before the report is written.
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                report=client.get(f'/api/comparisons/{id}/report')
                if report.status_code==200:break
                time.sleep(.05)
            self.assertEqual(report.status_code,200)
            self.assertEqual(report.json()['engines'][0],'matchanything-eloftr')
