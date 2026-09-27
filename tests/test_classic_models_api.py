import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
import server


class ClassicModelsAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        for target,folder in [('UPLOAD_DIR','uploads'),('PROJECTS_DIR','projects')]:
            p=patch.object(server,target,root/folder);p.start();self.addCleanup(p.stop)
        self.client=TestClient(server.app)
        rng=np.random.default_rng(17)
        scene=cv2.GaussianBlur(rng.integers(0,256,(200,500,3),dtype=np.uint8),(3,3),0)
        self.files=[('files',(f'tile_r00_c{i:02d}.png',cv2.imencode('.png',image)[1].tobytes(),'image/png'))
                    for i,image in enumerate([scene[:,:330],scene[:,170:]])]

    def test_reject_unknown_blocked_and_invalid_options_before_job(self):
        for engine,options in [('bogus',{}),('dad',{}),('sift',{'match_threshold':.2}),('sift',{'version':8}),('sift',[]),('sift',{'max_keypoints':-2})]:
            count=len(server.jobs)
            r=self.client.post('/api/stitch',files=self.files,data={'engine':engine,'engine_options':json.dumps(options)})
            self.assertEqual(r.status_code,400,r.text)
            self.assertEqual(len(server.jobs),count)

    def test_grid_artifacts_retry_save_reload_and_legacy_project(self):
        r=self.client.post('/api/stitch',files=self.files,data={'engine':'sift','alignment':'translation',
            'engine_options':json.dumps({'version':1,'max_keypoints':1500,'ratio_test':.7})})
        self.assertEqual(r.status_code,200,r.text)
        id=r.json()['job_id']; job=server.jobs[id]
        deadline=time.monotonic()+30
        while not job.done and time.monotonic()<deadline: time.sleep(.05)
        self.assertTrue(job.done)
        self.assertIsNotNone(job.result_path,[e for e in job.events if e['type']=='error'])
        diagnostic=next(e['diagnostic'] for e in job.events if e['type']=='pair_diagnostic')
        self.assertEqual(self.client.get(diagnostic['artifact']).status_code,200)
        for url in diagnostic['previews'].values(): self.assertEqual(self.client.get(url).status_code,200)
        before=len(job.events)
        retry=self.client.post('/api/pairs/compare',json={'job_id':id,'pair':[1,2],'engines':['sift','orb']})
        self.assertEqual(retry.status_code,200,retry.text)
        self.assertEqual(len(retry.json()['results']),2)
        self.assertEqual(len(job.events),before)
        self.assertTrue(retry.json()['results'][0]['accepted'])
        self.assertEqual(self.client.get(retry.json()['results'][0]['artifact']).status_code,200)
        saved=self.client.post('/api/projects',json={'job_id':id,'name':'Model test'}).json()
        project=self.client.get('/api/projects/'+saved['id']).json()
        self.assertEqual(project['meta']['options']['engine_options']['max_keypoints'],1500)
        self.assertEqual(project['meta']['options']['alignment'],'translation')
        loaded=next(e['diagnostic'] for e in project['events'] if e['type']=='pair_diagnostic')
        self.assertIn('/api/projects/',loaded['artifact'])
        self.assertEqual(self.client.get(loaded['artifact']).status_code,200)
        r=self.client.post('/api/pairs/compare',json={'project_id':saved['id'],'pair':[1,2],'engines':['sift']})
        self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(r.json()['results'][0]['accepted'])
        # Migration preserves historical settings without requiring weights or inference.
        meta_path=server.PROJECTS_DIR/saved['id']/'meta.json'
        meta=json.loads(meta_path.read_text());meta['options'].pop('engine_options');meta_path.write_text(json.dumps(meta))
        reloaded=self.client.get('/api/projects/'+saved['id']).json()
        self.assertEqual(reloaded['meta']['options']['engine_options']['version'],1)

if __name__=='__main__': unittest.main()
