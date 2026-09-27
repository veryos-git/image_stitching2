import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from stitcher import PanoramaStitcher, StitchConfig, StitchError
from stitcher.grid import DEFAULT_TEMPLATE, compile_template, inspect_names, require_positions


class GridTests(unittest.TestCase):
    def test_default_template(self):
        self.assertEqual(require_positions(['tile_r00_c09.png', 'sample_scan_r12_c103.PNG']),
                         [(0, 9), (12, 103)])

    def test_custom_templates(self):
        self.assertEqual(require_positions(['scan_row2_col17.tif'], 'scan_row{row}_col{col}.tif'), [(2, 17)])
        self.assertEqual(require_positions(['img_c03_r01.jpg'], 'prefix_{cNN}_{rNN}.jpg'), [(1, 3)])
        self.assertEqual(require_positions(['a+[r2]-c4.png'], 'a+[{rNN}]-{cNN}.png'), [(2, 4)])

    def test_invalid_names_duplicates_and_templates(self):
        result = inspect_names(['tile_r00_c09.png', 'tile_r0_c9.png', 'random.png'])
        self.assertFalse(result['valid'])
        self.assertEqual(len(result['errors']), 2)
        for name in ('tile_r00.png', 'tile_r00_c09Xpng', 'tile_r00_c09.png.bak'):
            with self.assertRaisesRegex(ValueError, 'separate Pipeline comparison'):
                require_positions([name])
        for template in ('prefix.png', '{row}_{row}_{col}.png', '{unknown}_{col}.png'):
            with self.assertRaises(ValueError):
                compile_template(template)

    def test_shuffled_two_dimensional_grid_matches_only_neighbors(self):
        rng = np.random.default_rng(18)
        canvas = cv2.GaussianBlur(rng.integers(0, 256, (600, 850, 3), dtype=np.uint8), (3, 3), 0)
        positions = [(1, 1), (0, 0), (1, 0), (0, 1)]
        images = [canvas[r*250:r*250+350, c*350:c*350+500].copy() for r, c in positions]
        config = StitchConfig(engine='sift', alignment='translation', pairing='grid',
                              grid_positions=positions, disconnected='reject', max_keypoints=4000,
                              viz=False, crop=False)
        result = PanoramaStitcher(config).stitch(images)
        graph = result.stats['graph']
        self.assertEqual(graph['mode'], 'grid')
        self.assertEqual(graph['tested_pairs'], 4)
        self.assertEqual(graph['accepted_pairs'], 4)
        self.assertEqual(result.panorama.shape[:2], (640, 890))
        for pair in graph['pairs']:
            a, b = [positions[i-1] for i in pair['pair']]
            self.assertEqual(abs(a[0]-b[0]) + abs(a[1]-b[1]), 1)

    def test_incremental_api_rejects_missing_duplicate_and_non_neighbor_positions(self):
        from server import app, incremental_sessions
        image = cv2.imencode('.png', np.zeros((32, 32, 3), np.uint8))[1].tobytes()
        def upload(name):
            return {'file': (name, image, 'image/png')}
        with TestClient(app) as client, patch('server.IncrementalStitcher') as factory, patch('server._save_map'):
            factory.return_value.count = 1
            invalid = client.post('/api/incremental/start', files=upload('random.png'))
            self.assertEqual(invalid.status_code, 400)
            factory.assert_not_called()
            response = client.post('/api/incremental/start', files=upload('tile_r00_c00.png'),
                                   data={'alignment':'translation'})
            self.assertTrue(response.json()['ok'], response.text)
            sid = response.json()['session_id']
            self.assertEqual(factory.call_args.args[0].alignment, 'translation')
            for name in ('random.png', 'tile_r00_c00.png', 'tile_r00_c02.png'):
                result = client.post('/api/incremental/add', data={'session_id':sid}, files=upload(name))
                self.assertFalse(result.json()['ok'], result.text)
            factory.return_value.add.assert_not_called()
            result = client.post('/api/incremental/add', data={'session_id':sid}, files=upload('tile_r00_c01.png'))
            self.assertTrue(result.json()['ok'], result.text)
            self.assertEqual(incremental_sessions[sid].grid_positions, {(0, 0), (0, 1)})
            incremental_sessions.pop(sid)

    def test_api_validation_and_custom_template(self):
        from server import app, jobs
        with TestClient(app) as client, patch('server._run_job'):
            preview = client.post('/api/grid/positions', json={'names':['tile_r00_c09.png'], 'template':DEFAULT_TEMPLATE})
            self.assertEqual(preview.json()['positions'], [[0, 9]])
            invalid = client.post('/api/grid/positions', json={'names':['random.png']})
            self.assertFalse(invalid.json()['valid'])
            self.assertIn('/compare', invalid.json()['help'])
            image = cv2.imencode('.png', np.zeros((32, 32, 3), np.uint8))[1].tobytes()
            files = [('files', ('scan_row0_col1.tif', image, 'image/png')),
                     ('files', ('scan_row0_col0.tif', image, 'image/png'))]
            self.assertEqual(client.post('/api/stitch', files=files).status_code, 400)
            response = client.post('/api/stitch', files=files,
                                   data={'filename_template':'scan_row{row}_col{col}.tif', 'alignment':'translation'})
            self.assertEqual(response.status_code, 200, response.text)
            config = jobs[response.json()['job_id']].config
            self.assertEqual(config['grid_positions'], [(0, 1), (0, 0)])
            self.assertEqual(config['pairing'], 'grid')
            self.assertEqual(config['disconnected'], 'reject')


if __name__ == '__main__':
    unittest.main()
