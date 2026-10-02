"""Resource gating tests mock only the external nvidia-smi command."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('training.a800_common'),
                             'explicit allocation gate is not implemented')
        self.common = importlib.import_module('training.a800_common')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def query(self, command, **kwargs):
        if '--query-gpu=index,uuid,name,memory.total' in command:
            return '0, GPU-test, NVIDIA A800 80GB PCIe, 81920\n'
        if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in command:
            return ''
        raise AssertionError(command)

    def gate(self, **kwargs):
        values = dict(gpu_index=0, session_minutes=30, allocation_confirmed=True, phase='train')
        values.update(kwargs)
        return self.common.require_allocation(self.root, **values)

    def test_missing_confirmation_never_queries_gpu_and_records_waiting(self):
        with patch('subprocess.check_output', side_effect=AssertionError('must not inspect GPU')):
            with self.assertRaises(self.common.AllocationUnavailable):
                self.gate(allocation_confirmed=False)
        status = json.loads((self.root/'outputs/progress/allocation-status.json').read_text())
        self.assertEqual(status['state'], 'waiting_for_allocation')

    def test_busy_gpu_refuses_without_touching_foreign_process(self):
        def busy(command, **kwargs):
            if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in command:
                return 'GPU-test, 987654, python, 1024\n'
            return self.query(command, **kwargs)
        with patch('subprocess.check_output', side_effect=busy):
            with self.assertRaisesRegex(self.common.AllocationUnavailable, 'occupied'):
                self.gate()

    def test_gate_preserves_deadline_and_rechecks_finalization(self):
        with patch('subprocess.check_output', side_effect=self.query):
            allocation = self.gate(phase='finalize')
            self.assertFalse(allocation.should_stop())
            self.assertAlmostEqual(allocation.remaining_seconds(), 1800, delta=2)
            self.assertEqual(allocation.recheck().gpu_uuid, 'GPU-test')
            self.assertEqual(allocation.phase, 'finalize')

    def test_invalid_or_reserve_only_duration_does_not_start_gpu(self):
        for value in (None, 0, -1, True, 2.5, 5):
            with self.subTest(minutes=value):
                with patch('subprocess.check_output', side_effect=AssertionError('no allocation')):
                    with self.assertRaises(self.common.AllocationUnavailable):
                        self.gate(session_minutes=value)

    def test_preflight_cannot_create_cuda_context_without_allocation(self):
        runner = importlib.import_module('training.a800_runner')
        with patch('torch.cuda.is_available', side_effect=AssertionError('CUDA queried before gate')):
            with self.assertRaises(self.common.AllocationUnavailable):
                runner.run_gpu_preflight(self.root, None)

    def test_changed_protocol_file_changes_source_identity(self):
        directory = self.root/'training/configs'
        directory.mkdir(parents=True)
        (directory/'protocol.json').write_text('{"version":1}')
        first = self.common.source_identity(self.root)
        (directory/'protocol.json').write_text('{"version":2}')
        second = self.common.source_identity(self.root)
        self.assertNotEqual(first, second)

    def test_symlink_or_parent_escape_is_not_a_bundle_input(self):
        for path in ('../outside', 'sealed/../../outside', 'C:/outside', '/outside', 'data\\file'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.common.bundle_path(self.root, path)

    def test_signal_stop_reaches_allocation_consumers_without_renewing_deadline(self):
        from dataclasses import replace
        with patch('subprocess.check_output', side_effect=self.query):
            original = self.gate()
            signaled = replace(original, stop_requested=lambda: True)
            self.assertTrue(signaled.should_stop())
            self.assertEqual(original.started_at, signaled.started_at)

    def test_training_entry_refuses_modified_bundle_before_cuda(self):
        from training.build_a800_bundle import write_integrity
        from training.a800_runner import main
        source = self.root/'training/registered.py'
        source.parent.mkdir()
        source.write_text('original')
        write_integrity(self.root, {'ready_for_upload': True})
        source.write_text('modified')
        with patch('subprocess.check_output', side_effect=self.query), patch(
                'torch.cuda.is_available', side_effect=AssertionError('CUDA used before bundle integrity gate')):
            with self.assertRaisesRegex(ValueError, 'Bundle payload mismatch'):
                main(['--root', str(self.root), '--phase', 'train', '--gpu', '0',
                      '--session-minutes', '30', '--allocation-confirmed'])
        self.assertFalse((self.root/'outputs/evidence/gpu-preflight.json').exists())

    def test_active_allocation_stops_when_disk_drops_below_checkpoint_reserve(self):
        with patch('subprocess.check_output', side_effect=self.query):
            allocation = self.gate()
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=19 * 1024**3)):
                self.assertTrue(allocation.should_stop())
                with self.assertRaisesRegex(self.common.AllocationUnavailable, 'disk'):
                    allocation.recheck()

    def test_low_disk_refuses_new_allocation_before_gpu_query(self):
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=19 * 1024**3)), patch(
                'subprocess.check_output', side_effect=AssertionError('no GPU work with exhausted storage')):
            with self.assertRaisesRegex(self.common.AllocationUnavailable, 'disk'):
                self.gate()

    def test_allocated_entry_refuses_unapproved_environment_before_cuda(self):
        from training.build_a800_bundle import write_integrity
        from training.a800_runner import main
        source = self.root/'training/registered.py'
        source.parent.mkdir()
        source.write_text('original')
        write_integrity(self.root, {'ready_for_upload': True})
        with patch('subprocess.check_output', side_effect=self.query), patch(
                'platform.system', return_value='Windows'), patch(
                'torch.cuda.is_available', side_effect=AssertionError('CUDA used before environment gate')):
            with self.assertRaisesRegex(RuntimeError, 'dedicated approved Linux prefix'):
                main(['--root', str(self.root), '--phase', 'train', '--gpu', '0',
                      '--session-minutes', '30', '--allocation-confirmed'])
        self.assertFalse((self.root/'outputs/evidence/gpu-preflight.json').exists())


if __name__ == '__main__':
    unittest.main()
