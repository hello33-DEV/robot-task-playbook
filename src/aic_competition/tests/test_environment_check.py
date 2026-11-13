import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('environment_check', ROOT/'tools/check_robot_environment.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class EnvironmentCheckTests(unittest.TestCase):
    def test_probe_uses_bounded_interpreter_without_shell(self):
        output = {'python': '2.7.17', 'executable': '/usr/bin/python2', 'imports': {}}
        response = types.SimpleNamespace(returncode=0, stdout='library message\n'+json.dumps(output)+'\n')
        with patch.object(CHECK.subprocess, 'run', return_value=response) as run:
            self.assertEqual(CHECK.inspect_python('/usr/bin/python2'), dict(output, ok=True))
        command = run.call_args[0][0]
        self.assertEqual(command[:3], ['/usr/bin/python2', '-B', '-c'])
        self.assertEqual(run.call_args[1]['timeout'], 8)
        self.assertNotIn('shell', run.call_args[1])

    def test_probe_errors_do_not_expose_payloads(self):
        for error in (OSError('private-value'), subprocess.TimeoutExpired('private-value', 8)):
            with self.subTest(error=type(error).__name__):
                with patch.object(CHECK.subprocess, 'run', side_effect=error):
                    report = CHECK.inspect_python('python')
                self.assertFalse(report['ok'])
                self.assertNotIn('private-value', json.dumps(report))

    def test_probe_rejects_failed_process_and_invalid_output(self):
        for returncode, stdout in ((1, 'private-value'), (0, ''), (0, 'invalid'), (0, '[]'), (0, '{}')):
            with self.subTest(returncode=returncode, stdout=stdout):
                response = types.SimpleNamespace(returncode=returncode, stdout=stdout)
                with patch.object(CHECK.subprocess, 'run', return_value=response):
                    self.assertFalse(CHECK.inspect_python('python')['ok'])

    def test_aliases_are_probed_once_and_missing_python2_is_reported(self):
        paths = {'python': '/same/python', 'python2': None, 'python3': '/same/python'}
        with patch.object(CHECK.shutil, 'which', side_effect=paths.get):
            with patch.object(CHECK, 'inspect_python', return_value={'ok': True}) as inspect:
                report = CHECK.compare_python()
        inspect.assert_called_once_with('/same/python')
        self.assertFalse(report['python2']['available'])
        self.assertTrue(report['python3']['available'])

    def test_os_release_only_returns_expected_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'os-release'
            path.write_text('ID=ubuntu\nVERSION_ID="18.04"\nPRETTY_NAME="Ubuntu 18.04"\nSECRET=private-value\n', encoding='utf-8')
            self.assertEqual(CHECK.os_release(path),
                             {'ID': 'ubuntu', 'VERSION_ID': '18.04', 'PRETTY_NAME': 'Ubuntu 18.04'})
            self.assertEqual(CHECK.os_release(Path(folder)/'missing'), {})

    def run_main(self, argv, module_error=None):
        stream = io.StringIO()
        environment = {'AIC_VISION_API_KEY': 'private-value', 'AIC_VISION_MODEL': 'private-value'}
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(CHECK.shutil, 'which', return_value=None):
                with patch.object(CHECK.importlib, 'import_module', side_effect=module_error,
                                  return_value=types.SimpleNamespace()):
                    with patch.object(CHECK, 'compare_python', return_value={'python2': {'available': False}}) as compare:
                        with contextlib.redirect_stdout(stream):
                            status = CHECK.main(argv)
        self.assertNotIn('private-value', stream.getvalue())
        return status, json.loads(stream.getvalue()), compare

    def test_default_reports_presence_not_credentials_and_preserves_exit_status(self):
        status, report, compare = self.run_main([])
        self.assertEqual(status, 0)
        self.assertTrue(report['credentials_present']['AIC_VISION_API_KEY'])
        self.assertFalse(report['credentials_present']['AIC_TTS_TOKEN'])
        self.assertIn('/dev/abot', report['serial_paths'])
        compare.assert_not_called()

    def test_missing_current_imports_are_not_hidden_by_comparison(self):
        status, report, compare = self.run_main(['--compare-python'], ImportError('private-value'))
        self.assertEqual(status, 1)
        self.assertFalse(report['imports']['rospy']['ok'])
        self.assertIn('python_runtimes', report)
        compare.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
