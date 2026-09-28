import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from stitcher import PanoramaStitcher, StitchConfig, StitchError
from stitcher.contract import prepare_image
from stitcher.enhancements import correct_flatfield, crop_matches, predicted_shift, solve_translations
from stitcher.geometry import translation_matrix
from stitcher.matching import build_engine
from stitcher.diagnostics import evaluate_pair


class EnhancementTests(unittest.TestCase):
    def config(self, **options):
        return StitchConfig(engine='sift', alignment='translation', pairing='grid',
                            grid_positions=[(0,0), (0,1), (1,0), (1,1)],
                            max_keypoints=4000, ransac_thresh=3, viz=False, crop=False,
                            exposure=False, disconnected='reject', **options)

    def tiles(self):
        scene = cv2.GaussianBlur(np.random.default_rng(41).integers(0,256,(480,680,3),dtype=np.uint8),(3,3),0)
        return [scene[y:y+300,x:x+420].copy() for y,x in [(0,0),(0,260),(180,0),(180,260)]]

    def test_global_solve_reconciles_loop_and_respects_weak_edge(self):
        edges = [(30,0,1,translation_matrix(-100,0)),
                 (30,0,2,translation_matrix(0,-100)),
                 (30,1,3,translation_matrix(0,-100)),
                 (30,2,3,translation_matrix(-104,0))]
        positions, groups = solve_translations(4, edges)
        residuals = [np.linalg.norm(positions[j]-positions[i]+H[:2,2]) for _,i,j,H in edges]
        np.testing.assert_allclose(residuals, 1., atol=1e-6)
        self.assertEqual(len(set(groups)), 1)
        edges[-1] = (.02,2,3,translation_matrix(-170,0))
        positions, _ = solve_translations(4, edges)
        np.testing.assert_allclose(positions[3], [100,100], atol=.1)

    def test_disconnected_positions_are_not_used_as_layout_predictions(self):
        edges = [(20,0,1,translation_matrix(-100,0))]
        positions, labels = solve_translations(3,edges)
        shift, _ = predicted_shift(1,2,edges,[(0,0),(0,1),(0,2)],positions,labels)
        self.assertIsNone(shift)  # One measured step is insufficient for a prior.

    def test_no_measurements_cannot_be_rescued_by_priors(self):
        cfg = self.config(global_adjustment=True, guided_retry=True, grid_priors=True)
        with self.assertRaises(StitchError):
            PanoramaStitcher(cfg).stitch([np.zeros((120,160,3),np.uint8) for _ in range(4)])

    def test_flatfield_reduces_stationary_shading(self):
        x = np.linspace(.5,1.5,240)
        image = np.tile((100*x).astype(np.uint8)[None,:,None], (160,1,3))
        corrected = correct_flatfield([image.copy() for _ in range(6)])
        self.assertLess(float(corrected[0].std()), float(image.std())*.2)
        self.assertEqual(corrected[0].dtype, np.uint8)
        with self.assertRaisesRegex(ValueError, 'equal-sized'):
            correct_flatfield([image, image[:80]])

    def test_guided_crop_returns_full_tile_coordinates(self):
        images = self.tiles()
        cfg = self.config()
        engine = build_engine(cfg.to_dict(), device='cpu')
        features = [prepare_image(engine,im,str(i+1)) for i,im in enumerate(images)]
        matches = crop_matches(engine,images,0,1,np.array([-260.,0]),8.)
        H, diagnostic, _ = evaluate_pair(engine,*features[:2],cfg,[1,2],matches=matches,
                                          predicted_shift=np.array([-260.,0]),prediction_radius=8.)
        self.assertTrue(diagnostic['accepted'])
        np.testing.assert_allclose(H[:2,2],[-260,0],atol=.2)
        H, diagnostic, _ = evaluate_pair(engine,*features[:2],cfg,[1,2],matches=matches,
                                          predicted_shift=np.array([-220.,0]),prediction_radius=8.)
        self.assertIsNone(H)
        self.assertIn('disagrees', diagnostic['reason'])

    def test_retry_recovers_failed_edge_and_keeps_separate_artifacts(self):
        original = evaluate_pair
        def fail_initial(engine,a,b,cfg,pair,**kw):
            H, d, preview = original(engine,a,b,cfg,pair,**kw)
            if pair == [1,2] and not kw:
                H = None
                d.update(accepted=False, reason='Synthetic initial failure')
            return H,d,preview
        with tempfile.TemporaryDirectory() as tmp, patch('stitcher.diagnostics.evaluate_pair',side_effect=fail_initial):
            cfg = self.config(guided_retry=True,global_adjustment=True,diagnostics_dir=tmp)
            result = PanoramaStitcher(cfg).stitch(self.tiles())
            graph = result.stats['graph']
            self.assertEqual(graph['accepted_pairs'],4)
            self.assertTrue(graph['guided_retries'][0]['accepted'])
            self.assertTrue((Path(tmp)/'guided/1-2.npz').exists())
            saved = json.loads((Path(tmp)/'graph.json').read_text())
            self.assertEqual(len(saved['positions']),4)
            self.assertFalse(saved['placement_includes_predictions'])

    def test_priors_are_opt_in_and_reported_without_inventing_measurements(self):
        # A 1x4 chain with one failed link and two agreeing measured steps.
        rng = np.random.default_rng(17)
        scene = cv2.GaussianBlur(rng.integers(0,256,(300,1200,3),dtype=np.uint8),(3,3),0)
        images = [scene[:,x:x+420].copy() for x in (0,260,520,780)]
        original = evaluate_pair
        def fail_middle(engine,a,b,cfg,pair,**kw):
            H,d,preview = original(engine,a,b,cfg,pair,**kw)
            if pair == [2,3]:
                H = None
                d.update(accepted=False,reason='Synthetic missing link')
            return H,d,preview
        cfg = self.config()
        cfg.grid_positions = [(0,i) for i in range(4)]
        with patch('stitcher.diagnostics.evaluate_pair',side_effect=fail_middle):
            with self.assertRaises(StitchError):
                PanoramaStitcher(cfg).stitch(images)
            cfg.grid_priors = cfg.global_adjustment = True
            result = PanoramaStitcher(cfg).stitch(images)
        graph = result.stats['graph']
        self.assertEqual(graph['accepted_pairs'],2)
        self.assertEqual(len(graph['predicted_edges']),1)
        self.assertFalse(graph['pairs'][1]['accepted'])
        self.assertTrue(graph['placement_includes_predictions'])

    def test_flatfield_runs_once_before_rows(self):
        from stitcher.enhancements import correct_flatfield as original
        cfg = StitchConfig(engine='sift',stitching_mode='rows_first',alignment='translation',
                           flatfield=True,grid_positions=[(0,0),(0,1),(1,0),(1,1)],
                           max_keypoints=4000,viz=False,crop=False)
        with patch('stitcher.enhancements.correct_flatfield',wraps=original) as correction:
            result = PanoramaStitcher(cfg).stitch(self.tiles())
        self.assertEqual(correction.call_count,1)
        self.assertTrue(result.stats['flatfield'])

    def test_api_rejects_incompatible_modes_and_persists_options(self):
        from server import app, jobs, _save_project, PROJECTS_DIR
        import shutil
        files = [('files',(f'tile_r00_c{i}.png',cv2.imencode('.png',im)[1].tobytes(),'image/png'))
                 for i,im in enumerate(self.tiles()[:2])]
        with TestClient(app) as client, patch('server._run_job'):
            for option in ('global_adjustment','guided_retry','grid_priors'):
                self.assertEqual(client.post('/api/stitch',files=files,data={option:'true','engine':'sift'}).status_code,400)
            self.assertEqual(client.post('/api/stitch',files=files,data={'flatfield':'true','preprocessing':'sobel'}).status_code,400)
            options = dict(engine='sift',stitching_mode='grid',flatfield='true',global_adjustment='true',guided_retry='true',grid_priors='true')
            response = client.post('/api/stitch',files=files,data=options)
            self.assertEqual(response.status_code,200,response.text)
            job_id = response.json()['job_id']
            job = jobs[job_id]
            job.done = True
            guided = Path(job.upload_dir)/'diagnostics/guided'
            guided.mkdir(parents=True)
            (guided/'1-2.json').write_text('{}')
            self.assertEqual(client.get(f'/api/jobs/{job_id}/diagnostics/guided/1-2.json').status_code,200)
            with patch('stitcher.enhancements.correct_flatfield', wraps=correct_flatfield) as correction:
                retry = client.post('/api/pairs/compare',json={'job_id':job_id,'pair':[1,2],'engines':['sift']})
                self.assertEqual(retry.status_code,200,retry.text)
                self.assertTrue(retry.json()['results'][0]['accepted'])
                self.assertEqual(correction.call_count,1)
            project = _save_project(job_id,'enhancement test')
            try:
                saved = client.get('/api/projects/'+project['id']).json()['meta']['options']
                for key in ('flatfield','global_adjustment','guided_retry','grid_priors'):
                    self.assertTrue(saved[key])
            finally:
                shutil.rmtree(PROJECTS_DIR/project['id'])
                job = jobs.pop(job_id)
                shutil.rmtree(job.upload_dir)


if __name__ == '__main__':
    unittest.main()
