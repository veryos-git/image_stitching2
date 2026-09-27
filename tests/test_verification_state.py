"""Readiness evidence follows source identity, not unrelated setup activity."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from stitcher.verification import environment_key


class VerificationStateTests(unittest.TestCase):
    def test_identical_source_rewrite_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory,patch('stitcher.verification.ROOT',Path(directory)):
            path=Path(directory)/'stitcher/adapter.py';path.parent.mkdir();path.write_text('value = 1\n')
            before=environment_key('sift')[0]
            path.write_text('value = 1\n')
            self.assertEqual(before,environment_key('sift')[0])
            path.write_text('value = 2\n')
            self.assertNotEqual(before,environment_key('sift')[0])

    def test_unrelated_vendor_setup_does_not_invalidate_xfeat(self):
        with tempfile.TemporaryDirectory() as directory,patch('stitcher.verification.ROOT',Path(directory)):
            before=environment_key('xfeat')[0]
            path=Path(directory)/'vendor/MatchAnything/src/model.py';path.parent.mkdir(parents=True);path.write_text('value = 1\n')
            self.assertEqual(before,environment_key('xfeat')[0])

if __name__=='__main__': unittest.main()
