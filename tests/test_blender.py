import unittest

import numpy as np

from stitcher import blender


class BlendTests(unittest.TestCase):
    def test_feather_smooths_overlap_and_preserves_uncovered_and_black_pixels(self):
        images = [np.full((40, 100, 3), value, np.uint8) for value in (40, 200)]
        masks = [np.zeros((40, 100), np.uint8) for _ in images]
        masks[0][:, :70] = 255
        masks[1][:, 30:90] = 255
        result = blender.feather_blend(images, masks)
        np.testing.assert_array_equal(result[:, :30], 40)
        np.testing.assert_array_equal(result[:, 70:90], 200)
        np.testing.assert_array_equal(result[:, 90:], 0)
        transition = result[20, 29:71, 0].astype(float)
        self.assertTrue(np.all(np.diff(transition) >= 0))
        self.assertLess(np.max(np.diff(transition)), 15)
        images[0][:] = 0
        black = blender.feather_blend(images, masks)
        self.assertGreater(black[20, 50, 0], 0)
        self.assertLess(black[20, 50, 0], 200)

    def test_feather_preserves_identical_content_through_two_passes(self):
        image = np.random.default_rng(2).integers(0, 256, (50, 100, 3), dtype=np.uint8)
        masks = [np.zeros(image.shape[:2], np.uint8) for _ in range(2)]
        masks[0][:, :70] = 255
        masks[1][:, 30:] = 255
        row = blender.feather_blend([image, image], masks)
        result = blender.feather_blend([row, row], masks)
        np.testing.assert_array_equal(result, image)

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
