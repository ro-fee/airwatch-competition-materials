"""Independent regressions for upload publication and required CPU checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import build_a800_bundle as bundle


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


class BundlePublicationReviewTests(unittest.TestCase):
    def _unbound_evidence(self, root, tests=0):
        write_json(root / 'outputs/evidence/environment/cpu-checks.json', {
            'status': 'passed', 'unit_tests': tests, 'skipped_tests': 0,
            'claim_scope': 'CPU preparation check only',
        })
        write_json(root / 'environment/candidate-status.json', {
            'status': 'candidate_linux_metadata_closure_verified',
            'linux_markers_explicitly_evaluated': True,
            'server_runtime_verified': False,
        })

    def test_zero_test_report_cannot_publish_empty_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._unbound_evidence(root)
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                bundle.finalize(root)

    def test_unbound_success_cannot_certify_current_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._unbound_evidence(root, tests=100)
            source = root / 'training/a800_runner.py'
            source.parent.mkdir()
            source.write_text('changed after independent CPU checks', encoding='utf-8')
            # An aggregate count alone carries no identity of the inspected files.
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                bundle.finalize(root)

    def test_failed_recheck_cannot_reuse_prior_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._unbound_evidence(root, tests=100)
            write_json(root / 'outputs/evidence/environment/cpu-checks-failed.json', {
                'status': 'failed', 'message': 'Required real dataset count mismatch',
            })
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                bundle.finalize(root)

    def test_staging_cannot_overwrite_registered_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, target, environment = base / 'source', base / 'bundle', base / 'environment'
            for folder in (source / 'training', target / 'training', environment):
                folder.mkdir(parents=True)
            (source / 'training/a800_runner.py').write_text('replacement', encoding='utf-8')
            existing = target / 'training/a800_runner.py'
            existing.write_text('already staged', encoding='utf-8')
            with self.assertRaises((ValueError, FileExistsError, RuntimeError)):
                bundle.stage(source, target, environment)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'already staged')

    def test_missing_required_entry_points_cannot_be_silently_staged(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, target, environment = base / 'source', base / 'bundle', base / 'environment'
            source.mkdir()
            environment.mkdir()
            with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                bundle.stage(source, target, environment)


class RequiredCpuTestsReviewTests(unittest.TestCase):
    def test_zero_executed_tests_refuses_before_real_data_or_success_output(self):
        from training import a800_cpu_checks as checks
        import airwatch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests = root / 'tests'
            tests.mkdir()
            for name in ('data', 'data_integrity', 'bundle', 'runner', 'allocation',
                         'evaluation', 'objective'):
                (tests / f'test_a800_{name}.py').touch()
            for name in ('test_uav_multiscale_tcn', 'test_uav_a800_contract'):
                (tests / f'{name}.py').touch()
            with patch.object(airwatch, '__file__', str(root / 'airwatch/__init__.py')), \
                    patch.object(checks, 'verify_environment'), \
                    patch.object(checks.unittest.defaultTestLoader, 'loadTestsFromNames',
                                 return_value=unittest.TestSuite()), \
                    patch.object(checks, 'check_real_data', return_value={}) as real_data, \
                    patch.object(checks.importlib.metadata, 'version', return_value='fixture'), \
                    patch('sys.argv', ['a800_cpu_checks', '--root', str(root)]):
                with self.assertRaises(RuntimeError):
                    checks.main()
                real_data.assert_not_called()
            self.assertFalse((root / 'outputs/evidence/environment/cpu-checks.json').exists())


if __name__ == '__main__':
    unittest.main()
