import unittest

import cv2
import numpy as np

from stitcher import PanoramaStitcher, StitchConfig, StitchError, ProgressReporter
from stitcher.utils import draw_matches


class PairDiagnosticsTests(unittest.TestCase):
    def test_rejected_pairs_keep_previews_and_graph_after_failure(self):
        rng = np.random.default_rng(73)
        canvas = cv2.GaussianBlur(rng.integers(0, 256, (220, 550, 3), dtype=np.uint8), (3, 3), 0)
        images = [canvas[:, :350], canvas[:, 200:], np.zeros((220, 350, 3), np.uint8)]
        names = ['tile_r00_c00.png', 'tile_r00_c01.png', 'tile_r00_c02.png']
        events = []
        cfg = StitchConfig(engine='sift', pairing='grid', grid_positions=[(0, 0), (0, 1), (0, 2)],
                           input_names=names, disconnected='reject', viz=True, max_keypoints=2000)
        with self.assertRaises(StitchError):
            PanoramaStitcher(cfg, ProgressReporter(emit=events.append)).stitch(images)
        previews = [e for e in events if e.get('meta', {}).get('pair_diagnostic')]
        self.assertEqual(len(previews), 2)
        self.assertTrue(previews[0]['meta']['pair_diagnostic']['accepted'])
        rejected = previews[1]
        self.assertEqual(rejected['meta']['names'], names[1:])
        self.assertEqual(rejected['meta']['pair_diagnostic']['matches'], 0)
        self.assertEqual(rejected['meta']['drawn_matches'], 0)
        self.assertIn('fewer than 12', rejected['meta']['pair_diagnostic']['reason'])
        self.assertTrue(rejected['image'].startswith('data:image/'))
        graph = next(e['detail']['graph'] for e in events if e.get('detail', {}).get('graph'))
        self.assertEqual(graph['components'], [[1, 2], [3]])

    def test_zero_matches_still_shows_both_images(self):
        left, right = np.full((30, 40), 50, np.uint8), np.full((20, 35), 150, np.uint8)
        preview = draw_matches(left, right, np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(preview.shape, (30, 75, 3))
        np.testing.assert_array_equal(preview[:30, :40], 50)
        np.testing.assert_array_equal(preview[:20, 40:], 150)

if __name__ == '__main__':
    unittest.main()
