import unittest
import numpy as np
from stitcher.translation import reliable_translation


class TranslationTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.linspace(80,320,20),np.linspace(80,320,20))
        self.points=np.column_stack((x.ravel(),y.ravel()))

    def test_offsets_with_outliers(self):
        rng=np.random.default_rng(5)
        for shift in ([50,-40],[-45,60]):
            target=self.points+shift+rng.normal(0,.3,self.points.shape)
            target[:80]=rng.uniform(0,400,(80,2))
            offset,count=reliable_translation(self.points,target,(400,400),(400,400))
            np.testing.assert_array_equal(offset,shift)
            self.assertGreaterEqual(count,320)

    def test_reject_rotation_scale_shear_perspective(self):
        p=self.points
        angle=.2
        for target in (
            (p-200)@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])+200,
            (p-200)*1.2+200,
            p@np.array([[1,.2],[0,1]]),
            p/(1+.001*p[:,0,None]),
        ):
            offset,_=reliable_translation(p,target,(400,400),(400,400))
            self.assertIsNone(offset)

    def test_reject_no_overlap_and_local_cluster(self):
        for p,target in ((self.points,self.points+[500,0]),(self.points/50,self.points/50+[50,0])):
            self.assertIsNone(reliable_translation(p,target,(400,400),(400,400))[0])
