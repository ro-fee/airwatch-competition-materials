"""Resume tests exercise real optimizer, dropout, data order and atomic files."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class TinyDataset:
    split = 'train'
    label_map = {'a': 0, 'b': 1, 'c': 2}
    data_identity = {'artifact_type': 'synthetic_engineering_fixture', 'sha256': 'a' * 64}

    def __init__(self):
        self.data = torch.arange(6 * 2 * 64, dtype=torch.float32).reshape(6, 2, 64) / 1000
        self.samples = [SimpleNamespace(window_id=f'w{i}', label_index=i % 3) for i in range(6)]

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return self.data[index].clone(), index % 3, index

    def sample_metadata(self, index):
        return self.samples[index]


def tiny_model():
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(128, 8),
                               torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(8, 3))


def actual_validation(model, dataset, device, **kwargs):
    model.eval()
    with torch.no_grad():
        logits = model(dataset.data.to(device))
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(6, device=device) % 3).item()
    return {'rank': [-loss, -loss, -loss, -loss], 'loss': loss}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('training.a800_runner'),
                             'resumable A800 runner is not implemented')
        self.runner = importlib.import_module('training.a800_runner')
        self.config = self.runner.make_run_config('D', 20260909)
        self.config.update(kind='engineering_smoke', run_id='engineering-smoke-resume')
        self.config['training'].update(anchors_per_batch=2, steps_per_epoch=3,
                                      logical_epochs=3, total_steps=9, warmup_steps=3,
                                      workers=0, precision='fp32', cpu_threads=1)
        self.code = {'test-fixture.py': 'b' * 64}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_core(self, name, **kwargs):
        return self.runner.run_training(self.config, TinyDataset(), TinyDataset(),
            self.root / name, device='cpu', code_identity=self.code,
            model_factory=lambda config: tiny_model(), evaluator=actual_validation, **kwargs)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_interruption_recovers_exact_samples_views_learning_rates_and_weights(self):
        self.run_core('continuous')
        partial = self.run_core('resumed', stop_after_steps=4)
        self.assertEqual(partial['state'], 'interrupted')
        self.assertEqual(partial['global_step'], 4)
        self.run_core('resumed')
        full = self.runner.load_checkpoint(self.root / 'continuous/last.pt')
        resumed = self.runner.load_checkpoint(self.root / 'resumed/last.pt')
        for field in ('model_state', 'optimizer_state', 'scheduler_state', 'rng', 'sampler',
                      'history', 'global_step', 'draw_position', 'best_state', 'best_rank', 'best_step'):
            with self.subTest(field=field):
                self.assert_nested_equal(full[field], resumed[field])
        self.assertEqual(resumed['global_step'], 9)
        self.assertEqual(resumed['draw_position'], 18)
        self.assertIsNone(resumed['scaler'])

    def test_resume_rejects_data_or_code_identity_change_without_overwriting(self):
        self.run_core('resume', stop_after_steps=2)
        before = (self.root / 'resume/last.pt').read_bytes()
        self.code['test-fixture.py'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_core('resume')
        self.assertEqual(before, (self.root / 'resume/last.pt').read_bytes())

    def test_corrupt_latest_recovers_previous_complete_checkpoint(self):
        self.run_core('resume', stop_after_steps=4)
        previous = self.runner.load_checkpoint(self.root / 'resume/last.previous.pt')
        self.assertEqual(previous['global_step'], 3)
        (self.root / 'resume/last.pt').write_bytes(b'incomplete pickle')
        self.run_core('resume')
        self.run_core('continuous')
        for field in ('model_state', 'history'):
            self.assert_nested_equal(self.runner.load_checkpoint(self.root / 'resume/last.pt')[field],
                                     self.runner.load_checkpoint(self.root / 'continuous/last.pt')[field])

    def test_formal_configuration_cannot_shorten_budget_or_use_cpu(self):
        formal = self.runner.make_run_config('A', 20260909)
        formal['training']['total_steps'] = 1
        with self.assertRaisesRegex(ValueError, 'formal|budget'):
            self.runner.validate_config(formal)
        formal = self.runner.make_run_config('A', 20260909)
        with self.assertRaisesRegex(ValueError, 'allocation|CUDA|cuda'):
            self.runner.run_training(formal, TinyDataset(), TinyDataset(), self.root/'bad', device='cpu')
        self.assertFalse((self.root/'bad/last.pt').exists())

    def test_sampling_repeats_complete_independent_permutations(self):
        order = self.runner.epoch_order(6, 24, seed=11, epoch=0)
        for start in range(0, 24, 6):
            self.assertEqual(sorted(order[start:start + 6].tolist()), list(range(6)))
        self.assertFalse(np.array_equal(order[:6], order[6:12]))

    def test_pending_validation_is_replayed_before_next_optimizer_step(self):
        calls = []
        def interrupted_validation(model, dataset, device, **kwargs):
            calls.append('validation')
            raise InterruptedError('allocated time ended during validation')
        result = self.runner.run_training(self.config, TinyDataset(), TinyDataset(),
            self.root/'pending', device='cpu', code_identity=self.code,
            model_factory=lambda config: tiny_model(), evaluator=interrupted_validation)
        self.assertEqual(result['state'], 'interrupted')
        checkpoint = self.runner.load_checkpoint(self.root/'pending/last.pt')
        self.assertEqual(checkpoint['global_step'], 3)
        self.assertTrue(checkpoint['validation_pending'])
        self.run_core('pending')
        self.run_core('continuous')
        for field in ('model_state', 'history', 'best_state'):
            self.assert_nested_equal(self.runner.load_checkpoint(self.root/'pending/last.pt')[field],
                                     self.runner.load_checkpoint(self.root/'continuous/last.pt')[field])

    def test_nan_run_is_failed_and_not_silently_retried(self):
        class BadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))
            def forward(self, values):
                return self.weight.unsqueeze(0).expand(len(values), -1) * float('nan')
        with self.assertRaises((FloatingPointError, ValueError)):
            self.runner.run_training(self.config, TinyDataset(), TinyDataset(), self.root/'nan',
                device='cpu', code_identity=self.code, model_factory=lambda config: BadModel(),
                evaluator=actual_validation)
        status = json.loads((self.root/'nan/run-status.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['global_step'], 0)
        with self.assertRaisesRegex(ValueError, 'failed run'):
            self.run_core('nan')

    def test_checkpoint_internal_cursor_tampering_is_rejected(self):
        self.run_core('resume', stop_after_steps=2)
        path = self.root/'resume/last.pt'
        state = self.runner.load_checkpoint(path)
        state['sampler']['position'] = 999
        torch.save(state, path)
        common = importlib.import_module('training.a800_common')
        common.atomic_json(path.with_suffix('.pt.sha256.json'),
            {'sha256': common.sha256_file(path), 'size_bytes': path.stat().st_size})
        with self.assertRaisesRegex(ValueError, 'cursor|sampler'):
            self.runner.load_checkpoint(path)

    def test_incomplete_recovery_state_uses_previous_checkpoint(self):
        self.run_core('resume', stop_after_steps=4)
        path = self.root/'resume/last.pt'
        state = self.runner.load_checkpoint(path)
        del state['timing']
        torch.save(state, path)
        common = importlib.import_module('training.a800_common')
        common.atomic_json(path.with_suffix('.pt.sha256.json'),
            {'sha256': common.sha256_file(path), 'size_bytes': path.stat().st_size})
        with self.assertRaisesRegex(ValueError, 'incomplete checkpoint'):
            self.runner.load_checkpoint(path)
        self.run_core('resume')
        self.run_core('continuous')
        for field in ('model_state', 'optimizer_state', 'history'):
            self.assert_nested_equal(self.runner.load_checkpoint(self.root/'resume/last.pt')[field],
                                     self.runner.load_checkpoint(self.root/'continuous/last.pt')[field])

    def test_cpu_smoke_does_not_initialize_cudnn_version(self):
        with patch('torch.backends.cudnn.version', side_effect=ValueError('no visible CUDA device')):
            result = self.run_core('cpu-only', stop_after_steps=1)
        self.assertEqual(result['state'], 'interrupted')
        checkpoint = self.runner.load_checkpoint(self.root/'cpu-only/last.pt')
        self.assertIsNone(checkpoint['identity']['environment']['cudnn'])

    def test_preflight_resume_check_serializes_and_compares_real_training(self):
        self.assertTrue(hasattr(self.runner, 'verify_resume_equivalence'))
        result = self.runner.verify_resume_equivalence(self.config, TinyDataset(), TinyDataset(),
            self.root/'resume-check', device='cpu', code_identity=self.code,
            model_factory=lambda config: tiny_model())
        self.assertEqual(result['state'], 'completed')
        self.assertTrue(result['exact_equal'])
        self.assertFalse(result['formal_result'])
        self.assertEqual(result['continuous_steps'], 4)
        self.assertEqual(result['interruption_step'], 2)
        self.assertEqual([(row['state'], row['global_step']) for row in result['run_results']],
                         [('completed', 4), ('interrupted', 2), ('completed', 4)])
        for name in ('continuous', 'resumed'):
            checkpoint = self.runner.load_checkpoint(self.root/'resume-check'/name/'last.pt')
            self.assertEqual(checkpoint['global_step'], 4)
            self.assertEqual(checkpoint['kind'], 'engineering_smoke')
            self.assertEqual(checkpoint['draw_position'], 12)
        evidence = json.loads((self.root/'resume-check/resume-equivalence.json').read_text())
        self.assertEqual(evidence, result)

    def test_preflight_resume_check_records_mismatch_as_failure(self):
        self.assertTrue(hasattr(self.runner, 'verify_resume_equivalence'))
        calls = []
        def changing_architecture(config):
            calls.append(None)
            model = tiny_model()
            if len(calls) == 3:
                model[3].p = .8  # Resume restores weights, but cannot repair changed model behavior.
            return model
        with self.assertRaisesRegex(ValueError, 'resume equivalence'):
            self.runner.verify_resume_equivalence(self.config, TinyDataset(), TinyDataset(),
                self.root/'mismatch', device='cpu', code_identity=self.code,
                model_factory=changing_architecture)
        evidence = json.loads((self.root/'mismatch/resume-equivalence.json').read_text())
        self.assertEqual(evidence['state'], 'failed')
        self.assertFalse(evidence['exact_equal'])

    def test_preflight_resume_check_honors_stop_before_first_update(self):
        self.assertTrue(hasattr(self.runner, 'verify_resume_equivalence'))
        with self.assertRaises(InterruptedError):
            self.runner.verify_resume_equivalence(self.config, TinyDataset(), TinyDataset(),
                self.root/'stopped', device='cpu', code_identity=self.code,
                model_factory=lambda config: tiny_model(), stop_requested=lambda: True)
        evidence = json.loads((self.root/'stopped/resume-equivalence.json').read_text())
        self.assertEqual(evidence['state'], 'interrupted')

    def test_formal_gate_requires_both_architectures_resume_evidence(self):
        path = self.root/'outputs/evidence/gpu-preflight.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'state': 'completed', 'code': self.code, 'gpu_uuid': 'GPU-test',
            'models': [{'model': 'tcn'}, {'model': 'multiscale_tcn'}]}))
        with self.assertRaisesRegex(ValueError, 'preflight|resume'):
            self.runner._verify_gpu_preflight(self.root, SimpleNamespace(gpu_uuid='GPU-test'), self.code)

    def test_formal_matrix_file_contains_only_exact_full_budget_runs(self):
        configs = self.runner.load_matrix(Path(__file__).resolve().parents[1])
        self.assertEqual(len(configs), 12)
        self.assertEqual({run['training']['total_steps'] for run in configs}, {9200})
        self.assertEqual({run['training']['anchors_per_batch'] * run['training']['steps_per_epoch']
                          for run in configs}, {29440})

    def test_tied_checkpoints_keep_earliest_and_corrupt_best_is_rebuilt(self):
        def tied(model, dataset, device, **kwargs):
            return {'rank': [1., 1., 1., 0.]}
        args = dict(device='cpu', code_identity=self.code,
                    model_factory=lambda config: tiny_model(), evaluator=tied)
        self.runner.run_training(self.config, TinyDataset(), TinyDataset(), self.root/'tied',
                                 stop_after_steps=4, **args)
        (self.root/'tied/best.pt').write_bytes(b'broken selected weight')
        self.runner.run_training(self.config, TinyDataset(), TinyDataset(), self.root/'tied', **args)
        last = self.runner.load_checkpoint(self.root/'tied/last.pt')
        best = self.runner.load_checkpoint(self.root/'tied/best.pt')
        self.assertEqual(last['best_step'], 3)
        self.assertEqual(best['global_step'], 3)
        self.assert_nested_equal(best['model_state'], last['best_state'])

    def test_train_intake_rejects_foreign_member_even_when_array_hashes_match(self):
        row = dict(recording_id='train-one', archive_id='source', label_index='0', member_index='0',
                   member_path='source/one.mat', member_sha256='a'*64, split='train')
        sample = SimpleNamespace(**row, window_id='w0')
        dataset = SimpleNamespace(samples=[sample], split='train', recording_ids={'train-one'})
        self.assertTrue(hasattr(self.runner, 'validate_dataset_assignments'))
        self.runner.validate_dataset_assignments(dataset, {'train-one': row}, windows_per_recording=1)
        sample.member_path = 'source/held-out.mat'
        with self.assertRaisesRegex(ValueError, 'provenance|assignment'):
            self.runner.validate_dataset_assignments(dataset, {'train-one': row}, windows_per_recording=1)

    def test_train_intake_rejects_omitted_recording_and_duplicate_window(self):
        self.assertTrue(hasattr(self.runner, 'validate_dataset_assignments'))
        first = dict(recording_id='r1', archive_id='a', label_index='0', member_index='0',
                     member_path='a/0.mat', member_sha256='a'*64, split='train')
        second = dict(first, recording_id='r2', member_index='1', member_path='a/1.mat')
        sample = SimpleNamespace(**first, window_id='w')
        dataset = SimpleNamespace(samples=[sample], split='train', recording_ids={'r1'})
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.runner.validate_dataset_assignments(dataset, {'r1': first, 'r2': second}, windows_per_recording=1)
        dataset.samples.append(sample)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.runner.validate_dataset_assignments(dataset, {'r1': first}, windows_per_recording=2)


if __name__ == '__main__':
    unittest.main()
