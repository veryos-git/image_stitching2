import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from stitcher import PanoramaStitcher, IncrementalStitcher, StitchConfig
from stitcher.preprocessing import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_width_preserves_aspect_and_does_not_upscale_or_modify_source(self):
        image = np.full((800, 1600, 3), 80, np.uint8)
        self.assertEqual(preprocess_image(image, 1080).shape, (540, 1080, 3))
        self.assertIs(preprocess_image(image, 0), image)
        self.assertIs(preprocess_image(image, 1920), image)
        np.testing.assert_array_equal(image, 80)
        self.assertEqual(preprocess_image(np.zeros((1600, 800, 3), np.uint8), 400).shape,
                         (800, 400, 3))

    def test_sobel_detects_both_axes_and_preserves_zero_regions(self):
        image = np.zeros((80, 80, 3), np.uint8)
        image[20:60, 20:60] = 255
        edges = preprocess_image(image, 40, 'sobel')
        self.assertEqual(edges.shape, (40, 40, 3))
        self.assertEqual(edges.dtype, np.uint8)
        self.assertGreater(edges[10, 20, 0], 0)
        self.assertGreater(edges[20, 10, 0], 0)
        self.assertEqual(edges[20, 20, 0], 0)
        np.testing.assert_array_equal(edges[:, :, 0], edges[:, :, 1])
        np.testing.assert_array_equal(edges[:, :, 1], edges[:, :, 2])
        np.testing.assert_array_equal(preprocess_image(np.full_like(image, 100), 0, 'sobel'), 0)

    def test_rows_preprocess_original_tiles_once_and_keep_wide_row_images(self):
        rng = np.random.default_rng(18)
        canvas = cv2.GaussianBlur(rng.integers(0, 256, (600, 850, 3), dtype=np.uint8), (3, 3), 0)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        images = [canvas[r*250:r*250+350, c*350:c*350+500].copy() for r, c in positions]
        cfg = StitchConfig(engine='sift', alignment='translation', stitching_mode='rows_first',
                           grid_positions=positions, input_max_width=250, preprocessing='sobel',
                           max_keypoints=4000, viz=False, exposure=False, crop=True)
        with patch('stitcher.preprocessing.preprocess_image', wraps=preprocess_image) as prep:
            result = PanoramaStitcher(cfg).stitch(images)
        self.assertEqual(prep.call_count, 4)
        self.assertFalse(result.stats['partial'])
        self.assertEqual(result.panorama.shape[:2], (300, 425))
        self.assertEqual(result.stats['preprocessing'], 'sobel')
        np.testing.assert_array_equal(result.panorama[:, :, 0], result.panorama[:, :, 1])
        with tempfile.TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f'{i}.png') for i in range(len(images))]
            for path, image in zip(paths, images):
                cv2.imwrite(path, image)
            with patch('stitcher.preprocessing.preprocess_image', wraps=preprocess_image) as prep:
                from_paths = PanoramaStitcher(cfg).stitch_paths(paths)
            self.assertEqual(prep.call_count, 4)
            np.testing.assert_array_equal(from_paths.panorama, result.panorama)

    def test_incremental_preprocesses_each_input_once(self):
        rng = np.random.default_rng(19)
        canvas = cv2.GaussianBlur(rng.integers(0, 256, (400, 850, 3), dtype=np.uint8), (3, 3), 0)
        cfg = StitchConfig(engine='sift', alignment='translation', input_max_width=250,
                           preprocessing='sobel', max_keypoints=4000, viz=False)
        stitcher = IncrementalStitcher(cfg)
        images = [canvas[:, :500], canvas[:, 350:]]
        with patch('stitcher.preprocessing.preprocess_image', wraps=preprocess_image) as prep:
            stitcher.seed(images[0])
            np.testing.assert_array_equal(stitcher.map_img, preprocess_image(images[0], 250, 'sobel'))
            stitcher.add(images[1])
        self.assertEqual(prep.call_count, 2)
        # Subpixel registration can add one pixel of canvas padding.
        np.testing.assert_allclose(stitcher.map_img.shape[:2], (200, 425), atol=1)
        np.testing.assert_allclose(np.asarray(stitcher.last_view) + stitcher.offset, [175, 0], atol=0.5)

    def test_api_defaults_options_and_rejection(self):
        import server
        files = [('files', (f'tile_r00_c{i:02d}.png', b'fixture', 'image/png')) for i in range(2)]
        with TestClient(server.app) as client, patch('server._run_job'):
            for options, width, mode in [({}, 1080, 'none'),
                                         ({'input_max_width': 720, 'preprocessing': 'sobel'}, 720, 'sobel'),
                                         ({'input_max_width': 0}, 0, 'none')]:
                response = client.post('/api/stitch', files=files, data={'engine': 'sift', **options})
                self.assertEqual(response.status_code, 200, response.text)
                cfg = server.jobs[response.json()['job_id']].config
                self.assertEqual(cfg['input_max_width'], width)
                self.assertEqual(cfg['preprocessing'], mode)
            for options in ({'input_max_width': -1}, {'input_max_width': 16385}, {'preprocessing': 'invalid'}):
                response = client.post('/api/stitch', files=files, data={'engine': 'sift', **options})
                self.assertEqual(response.status_code, 422, response.text)


if __name__ == '__main__':
    unittest.main()
