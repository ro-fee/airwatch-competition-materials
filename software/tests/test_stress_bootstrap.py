"""Frozen stress mode is explicit; normal desktop startup is unchanged."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stress_bootstrap_under_test',
    Path(__file__).resolve().parents[1]/'packaging/airwatch_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)

class StressBootstrapTests(unittest.TestCase):
    def test_explicit_stress_mode_returns_driver_exit_without_error_dialog(self):
        for code in (0, 1, 2):
            with self.subTest(code=code), patch.object(sys, 'argv',
                ['AirWatch.exe', '--ui-stress', '--output', 'evidence']), \
                patch.dict(sys.modules, {'main':SimpleNamespace(main=lambda:17)}), \
                patch('importlib.import_module', side_effect=SystemExit(code)) as load, \
                patch.object(bootstrap, '_show_error') as error:
                self.assertEqual(bootstrap.main(), code)
                load.assert_called_once_with('tools.ui_stress_session')
                self.assertEqual(sys.argv, ['AirWatch.exe', '--output', 'evidence'])
                error.assert_not_called()

    def test_normal_startup_never_loads_stress_driver(self):
        with patch.object(sys,'argv',['AirWatch.exe']), \
             patch.dict(sys.modules, {'main':SimpleNamespace(main=lambda:17)}), \
             patch('importlib.import_module') as load:
            self.assertEqual(bootstrap.main(),17)
            load.assert_not_called()

if __name__=='__main__': unittest.main()
