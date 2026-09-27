"""Launcher bootstrapping tests; no network or model downloads required."""
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class StartTests(unittest.TestCase):
    def setUp(self):
        self.launcher = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'start'))
        self.prepare = self.launcher['prepare_environment']
        self.globals = self.prepare.__globals__
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('start', *self.launcher['REQUIREMENTS'], *self.launcher['SETUP_SCRIPTS']):
            (self.root / name).write_text(name)
        self.globals['ROOT'] = self.root
        self.commands = []

    def fake_run(self, *command):
        self.commands.append(tuple(map(str, command)))
        # Stand in for successful package installs and model setup.
        for name in (
            '.venv/bin/python',
            'vendor/accelerated_features/modules/xfeat.py',
            'vendor/EfficientLoFTR/src/loftr/loftr.py',
            'vendor/MatchAnything/src/loftr/loftr.py',
            'weights/superpoint_v1.pth', 'weights/superglue_indoor.pth',
            'weights/superglue_outdoor.pth', 'weights/eloftr_outdoor.ckpt',
            'weights/matchanything_eloftr.ckpt',
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def prepare_mocked(self):
        with patch.dict(self.globals, run=self.fake_run), patch('shutil.which', return_value='/usr/bin/git'):
            with contextlib.redirect_stdout(io.StringIO()):
                return self.prepare()

    def test_fresh_setup_and_cached_restart(self):
        self.assertEqual(self.prepare_mocked(), self.root / '.venv/bin/python')
        self.assertIn('-m', self.commands[0])
        self.assertIn('venv', self.commands[0])
        install = next(c for c in self.commands if '-r' in c)
        for name in self.launcher['REQUIREMENTS']:
            self.assertIn(str(self.root / name), install)
        for name in self.launcher['SETUP_SCRIPTS']:
            self.assertIn((str(self.root / '.venv/bin/python'), str(self.root / name)), self.commands)
        self.commands.clear()
        self.prepare_mocked()
        self.assertEqual(self.commands, [])

    def test_changed_requirements_repeat_setup(self):
        self.prepare_mocked()
        self.commands.clear()
        (self.root / 'requirements.txt').write_text('changed')
        self.prepare_mocked()
        self.assertTrue(self.commands)
        self.assertFalse(any('venv' in c for c in self.commands))

    def test_missing_checkpoint_repeats_setup(self):
        self.prepare_mocked()
        self.commands.clear()
        (self.root / 'weights/eloftr_outdoor.ckpt').unlink()
        self.prepare_mocked()
        self.assertTrue(self.commands)

    def test_failure_does_not_cache_success(self):
        self.prepare_mocked()
        (self.root / 'requirements.txt').write_text('changed')
        def fail(*command):
            raise subprocess.CalledProcessError(1, command)
        with patch.dict(self.globals, run=fail), patch('shutil.which', return_value='/usr/bin/git'):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                self.prepare()
        self.assertFalse((self.root / '.venv/.stitch-lab-setup').exists())
        self.prepare_mocked()
        self.assertTrue((self.root / '.venv/.stitch-lab-setup').exists())

    def test_running_app_skips_setup(self):
        response = io.StringIO('{"app": "stitch-lab", "overview": "/"}')
        with patch('sys.argv', ['start']), patch.dict(self.globals) as namespace:
            with patch('urllib.request.OpenerDirector.open', return_value=response), patch('os.execv') as execute:
                namespace['prepare_environment'] = lambda: self.fail('Should reuse running app')
                with contextlib.redirect_stdout(io.StringIO()):
                    self.launcher['main']()
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
