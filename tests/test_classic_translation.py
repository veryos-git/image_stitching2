import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from stitcher import IncrementalStitcher, PanoramaStitcher, StitchConfig, StitchError
from stitcher import blender
from stitcher.unordered import reliable_edge


class ClassicTranslationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.canvas = cv2.GaussianBlur(
            rng.integers(0, 256, (320, 1200, 3), dtype=np.uint8), (3, 3), 0)
        self.images = [self.canvas[:, x:x + 500].copy() for x in (0, 350, 700)]

    def config(self, **kwargs):
        return StitchConfig(engine='sift', alignment='translation', refine=True,
                            feature_max_dim=800, max_keypoints=4000,
                            crop=False, viz=False, **kwargs)

    def test_batch_transforms_stay_translation_only(self):
        for pairing, images in [('sequential', self.images),
                                ('unordered', [self.images[2], self.images[0], self.images[1]])]:
            with self.subTest(pairing=pairing), \
                 patch('stitcher.stitcher.estimate_homography', side_effect=AssertionError('perspective fit')), \
                 patch('stitcher.unordered.estimate_homography', side_effect=AssertionError('perspective fit')), \
                 patch('stitcher.stitcher.refine_homography', side_effect=AssertionError('perspective refinement')), \
                 patch('stitcher.blender.warp_image', wraps=blender.warp_image) as warp:
                result = PanoramaStitcher(self.config(pairing=pairing)).stitch(images)
                self.assertEqual(result.stats['alignment'], 'translation')
                self.assertEqual(result.panorama.shape[:2], (360, 1240))
                for call in warp.call_args_list:
                    transform = call.args[1].copy()
                    transform[:2, 2] = 0
                    np.testing.assert_array_equal(transform, np.eye(3))

    def test_rejects_scale_without_fallback(self):
        x, y = np.meshgrid(np.linspace(60, 340, 20), np.linspace(60, 340, 20))
        points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        target = (points - 200) * 1.2 + 200
        H, _, reason = reliable_edge(points, target, (400, 400), (400, 400), 3, 'translation')
        self.assertIsNone(H)
        self.assertIn('translation-only', reason)
        self.assertIsNotNone(reliable_edge(points, target, (400, 400), (400, 400), 3)[0])
        with patch('stitcher.matching.ClassicEngine.match', return_value=(points, target, np.ones(len(points)))):
            with self.assertRaisesRegex(StitchError, 'translation-only'):
                PanoramaStitcher(self.config()).stitch(self.images[:2])

    def test_incremental_negative_world_origin(self):
        stitcher = IncrementalStitcher(self.config())
        with patch('stitcher.incremental.estimate_homography', side_effect=AssertionError('perspective fit')):
            stitcher.seed(self.images[1])
            stitcher.add(self.images[0])
            stitcher.add(self.images[2])
        np.testing.assert_array_equal(stitcher.offset, [-350, 0])
        self.assertEqual(stitcher.map_img.shape[:2], (320, 1200))

    def test_api_passes_alignment_and_rejects_invalid_values(self):
        from server import app, jobs
        files = [('files', (f'tile_r00_c{i:02d}.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png'))
                 for i, img in enumerate(self.images[:2])]
        with TestClient(app) as client, patch('server._run_job'):
            response = client.post('/api/stitch', data={'alignment': 'translation', 'engine': 'sift'}, files=files)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(jobs[response.json()['job_id']].config['alignment'], 'translation')
            self.assertEqual(client.post('/api/stitch', data={'alignment': 'invalid'}, files=files).status_code, 422)


if __name__ == '__main__':
    unittest.main()
