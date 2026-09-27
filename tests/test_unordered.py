import unittest
import cv2
import numpy as np
from stitcher import PanoramaStitcher, StitchConfig, StitchError
from stitcher.unordered import reliable_edge

class UnorderedTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.canvas=cv2.GaussianBlur(rng.integers(0,256,(320,1200,3),dtype=np.uint8),(3,3),0)
        self.images=[self.canvas[:,x:x+500].copy() for x in (0,350,700)]
    def stitch(self, images, disconnected='largest'):
        return PanoramaStitcher(StitchConfig(engine='sift', pairing='unordered', disconnected=disconnected, feature_max_dim=800, max_keypoints=4000, crop=False, viz=False, max_output_dim=1600)).stitch(images)
    def test_shuffled_chain(self):
        result=self.stitch([self.images[2],self.images[0],self.images[1]])
        graph=result.stats['graph']
        self.assertEqual(graph['included'],[1,2,3])
        self.assertEqual(graph['tested_pairs'],3)
        self.assertEqual(graph['accepted_pairs'],2)
        self.assertAlmostEqual(result.panorama.shape[1],1240,delta=5)
        self.assertAlmostEqual(result.panorama.shape[0],360,delta=5)
    def test_unconnected_image_partial_and_strict(self):
        unrelated=np.random.default_rng(501).integers(0,256,(320,500,3),dtype=np.uint8)
        images=[self.images[1],unrelated,self.images[0]]
        result=self.stitch(images)
        self.assertEqual(result.stats['graph']['included'],[1,3])
        self.assertEqual(result.stats['graph']['excluded'],[2])
        with self.assertRaisesRegex(StitchError,'disconnected'):
            self.stitch(images,'reject')
    def test_no_overlap(self):
        with self.assertRaisesRegex(StitchError,'No reliable overlap'):
            self.stitch([self.images[0],self.images[2]])
    def test_tiny_cluster_is_not_overlap(self):
        pts=np.random.default_rng(5).uniform(0,3,(30,2)).astype(np.float32)
        H,count,reason=reliable_edge(pts,pts+20,(1000,1000),(1000,1000),3)
        self.assertIsNone(H)
        self.assertIn('small region',reason)

if __name__=='__main__':unittest.main()
