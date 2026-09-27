import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from stitcher import PanoramaStitcher, StitchConfig, StitchError
from stitcher.progress import ProgressReporter


class RowsFirstTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18)
        self.canvas = cv2.GaussianBlur(rng.integers(0, 256, (600, 850, 3), dtype=np.uint8), (3, 3), 0)

    def stitch(self, positions):
        images = [self.canvas[r*250:r*250+350, c*350:c*350+500].copy() for r, c in positions]
        cfg = StitchConfig(engine='sift', alignment='translation', stitching_mode='rows_first',
                           grid_positions=positions, max_keypoints=4000, viz=False,
                           exposure=False, crop=True)
        rep = ProgressReporter()
        result = PanoramaStitcher(cfg, rep).stitch(images)
        return result, rep.history

    def test_two_stages_order_shuffled_tiles_and_emit_only_one_result(self):
        result, events = self.stitch([(1, 1), (0, 0), (1, 0), (0, 1)])
        self.assertEqual(result.panorama.shape[:2], (600, 850))
        # Two blend passes change intensity around seams; check spatial alignment.
        shift, response = cv2.phaseCorrelate(
            cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY).astype(np.float32),
            cv2.cvtColor(result.panorama, cv2.COLOR_BGR2GRAY).astype(np.float32))
        self.assertLess(np.linalg.norm(shift), 0.5)
        self.assertGreater(response, 0.6)
        self.assertEqual(result.stats['num_images'], 4)
        self.assertEqual(result.stats['num_rows'], 2)
        self.assertEqual([r['input_indices'] for r in result.stats['rows']], [[1, 3], [2, 0]])
        self.assertEqual(sum(e['type'] == 'plan' for e in events), 1)
        self.assertEqual(sum(e['type'] == 'result' for e in events), 1)
        steps = [e for e in events if e['type'] == 'step']
        progress = [e['progress'] for e in steps]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 1)
        row_done = [i for i, e in enumerate(events) if e['type'] == 'step'
                    and e['marker'].startswith('row_') and e['status'] == 'done']
        combine_start = next(i for i, e in enumerate(events) if e['type'] == 'step'
                             and e['marker'] == 'combine_rows')
        self.assertTrue(all(i < combine_start for i in row_done))

    def test_single_row_and_single_tile_rows(self):
        for positions, shape in [([(0, 1), (0, 0)], (350, 850)),
                                 ([(1, 0), (0, 0)], (600, 500))]:
            with self.subTest(positions=positions):
                result, _ = self.stitch(positions)
                self.assertEqual(result.panorama.shape[:2], shape)

    def test_requires_positions(self):
        cfg = StitchConfig(engine='sift', stitching_mode='rows_first')
        with self.assertRaisesRegex(StitchError, 'row and column'):
            PanoramaStitcher(cfg).stitch([self.canvas, self.canvas])

    def run_partial(self, images, positions):
        rep = ProgressReporter()
        cfg = StitchConfig(engine='sift', alignment='translation', stitching_mode='rows_first',
                           grid_positions=positions, max_keypoints=4000, viz=False,
                           exposure=False, crop=True)
        result = PanoramaStitcher(cfg, rep).stitch(images)
        self.assertTrue(result.stats['partial'])
        self.assertEqual(sum(e['type'] == 'result' for e in rep.history), 1)
        self.assertFalse(any(e['type'] == 'error' for e in rep.history))
        return result, rep.history

    def test_failed_row_does_not_prevent_later_rows_from_combining(self):
        images = [np.zeros((350, 500, 3), np.uint8)] * 2
        images += [self.canvas[:350].copy(), self.canvas[250:].copy()]
        result, events = self.run_partial(images, [(0, 0), (0, 1), (1, 0), (2, 0)])
        self.assertEqual(result.stats['included_rows'], [1, 2])
        self.assertEqual(result.stats['excluded_rows'], [0])
        self.assertEqual(result.stats['failed_rows'][0]['row'], 0)
        self.assertEqual(result.stats['num_images'], 2)
        self.assertEqual(result.stats['input_count'], 4)
        self.assertEqual(result.panorama.shape[:2], (600, 850))
        completed = [e['meta']['completed_row'] for e in events
                     if e['type'] == 'image' and 'completed_row' in e['meta']]
        self.assertEqual(completed, [1, 2])

    def test_disconnected_first_row_keeps_later_connected_rows(self):
        images = [np.zeros((350, 500, 3), np.uint8),
                  self.canvas[:350].copy(), self.canvas[250:].copy()]
        result, _ = self.run_partial(images, [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result.stats['included_rows'], [1, 2])
        self.assertEqual(result.stats['completed_rows'], [0, 1, 2])
        self.assertEqual(result.stats['failed_rows'], [])

    def test_no_row_overlap_returns_largest_completed_row(self):
        images = [np.zeros((100, 200, 3), np.uint8), self.canvas[:350].copy()]
        result, events = self.run_partial(images, [(0, 0), (1, 0)])
        self.assertEqual(result.stats['included_rows'], [1])
        self.assertEqual(result.stats['completed_rows'], [0, 1])
        np.testing.assert_array_equal(result.panorama, images[1])
        self.assertTrue(any('Could not combine rows' in w for w in result.stats['warnings']))
        self.assertEqual(sum(e['type'] == 'image' and 'completed_row' in e['meta'] for e in events), 2)

    def test_all_rows_failed_reports_failure_without_a_result(self):
        cfg = StitchConfig(engine='sift', alignment='translation', stitching_mode='rows_first',
                           grid_positions=[(0, 0), (0, 1)], viz=False)
        rep = ProgressReporter()
        with self.assertRaisesRegex(StitchError, 'No rows could be stitched'):
            PanoramaStitcher(cfg, rep).stitch([np.zeros((100, 100, 3), np.uint8)] * 2)
        self.assertFalse(any(e['type'] == 'result' for e in rep.history))

    def test_empty_row_borders_are_not_treated_as_image_content(self):
        images = [self.canvas[0:350, :500], self.canvas[10:360, 350:],
                  self.canvas[250:600, :500], self.canvas[250:600, 350:]]
        cfg = StitchConfig(engine='sift', alignment='translation', stitching_mode='rows_first',
                           grid_positions=[(0, 0), (0, 1), (1, 0), (1, 1)],
                           max_keypoints=4000, viz=False, exposure=False, crop=False)
        result = PanoramaStitcher(cfg).stitch(images)
        x, y, w, h = cv2.boundingRect(result.coverage)
        self.assertEqual((w, h), (850, 600))
        self.assertEqual(result.coverage[y + 2, x + w - 5], 0)
        self.assertEqual(result.coverage[y + 20, x + w - 5], 255)

    def test_api_defaults_and_mode_validation(self):
        from server import app, jobs
        image = cv2.imencode('.png', self.canvas[:32, :32])[1].tobytes()
        files = [('files', (f'tile_r00_c{i:02d}.png', image, 'image/png')) for i in range(2)]
        with TestClient(app) as client, patch('server._run_job'):
            for mode in ('grid', 'rows_first'):
                response = client.post('/api/stitch', files=files,
                                       data={'engine':'sift', 'stitching_mode':mode})
                self.assertEqual(response.status_code, 200, response.text)
                config = jobs[response.json()['job_id']].config
                self.assertEqual(config['alignment'], 'translation')
                self.assertEqual(config['stitching_mode'], mode)
            self.assertEqual(client.post('/api/stitch', files=files,
                                         data={'stitching_mode':'invalid'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
