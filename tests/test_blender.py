import unittest

import numpy as np

from stitcher import blender


class BlendTests(unittest.TestCase):
    def test_blend_preserves_constant_color_and_reports_work(self):
        images = [np.full((65, 79, 3), value, np.uint8) for value in (60, 120, 180)]
        masks = [np.full((65, 79), 255, np.uint8) for _ in images]
        events = []
        result = blender.multi_band_blend(
            images, masks, progress=lambda f, msg: events.append((f, msg)))
        np.testing.assert_array_equal(result, np.full_like(result, 120))
        fractions = [f for f, _ in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(any('image 3/3' in msg for _, msg in events))
        self.assertTrue(any('Reconstructing' in msg for _, msg in events))

    def test_exposure_uses_actual_overlap_and_skips_disjoint_masks(self):
        images = [np.full((80, 160, 3), value, np.uint8) for value in (60, 120, 200)]
        masks = [np.zeros((80, 160), np.uint8) for _ in images]
        masks[0][10:70, 5:65] = 255
        masks[1][10:70, 35:95] = 255
        masks[2][10:70, 120:150] = 255
        events = []
        gains = blender.exposure_compensation(
            images, masks, progress=lambda f, msg: events.append((f, msg)))
        np.testing.assert_allclose(gains, [np.sqrt(2), 1 / np.sqrt(2), 1], rtol=1e-6)
        self.assertEqual(events[-1][0], 1.0)
        self.assertEqual(len(events), 4)


if __name__ == '__main__':
    unittest.main()
