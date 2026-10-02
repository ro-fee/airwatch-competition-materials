"""CPU numerical fixtures; no formal experiment evidence is produced."""
import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch


class EvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.a800_evaluation'),
                             'The frozen evaluator must exist')
        from training import a800_evaluation
        return a800_evaluation

    def test_calibration_accepts_23_of_24_and_preserves_ties(self):
        api = self.api()
        result = api.calibrate_scores(np.arange(24) / 24, split='validation')
        self.assertEqual(result['threshold'], 1 / 24)
        self.assertEqual(result['accepted'], 23)
        tied = api.calibrate_scores(np.ones(24), split='validation')
        self.assertEqual(tied['accepted'], 24)
        for split in ['train', 'test', 'unknown', 'vti']:
            with self.assertRaises(ValueError):
                api.calibrate_scores(np.ones(24), split=split)

    def test_known_positive_roc_uses_steps_and_frozen_threshold(self):
        api = self.api()
        result = api.open_set_metrics([.8, .9], [.1, .85], .8)
        self.assertAlmostEqual(result['auroc'], .75)
        self.assertEqual(result['fpr95'], .5)
        self.assertEqual(result['known_acceptance'], 1)
        self.assertEqual(result['unknown_false_acceptance'], .5)
        self.assertEqual(result['threshold'], .8)

    def test_prototypes_are_member_weighted_normalized_and_train_v1_only(self):
        api = self.api()
        features = np.repeat([[2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], 32, axis=0)
        targets = np.repeat([0, 1, 2], 32)
        ids = np.repeat(['a', 'b', 'c'], 32)
        result = api.fit_prototypes(features, targets, ids, split='train', artifact_type='ku_leuven_materialized_known_train_validation')
        np.testing.assert_allclose(result, np.eye(3))
        for split, artifact in [('test', 'ku_leuven_materialized_known_train_validation'), ('train', 'ku_leuven_a800_v2')]:
            with self.assertRaises(ValueError):
                api.fit_prototypes(features, targets, ids, split=split, artifact_type=artifact)
        with self.assertRaises(ValueError):
            api.fit_prototypes(features[:-1], targets[:-1], ids[:-1], split='train', artifact_type='ku_leuven_materialized_known_train_validation')
        with self.assertRaises(ValueError):
            api.fit_prototypes(features * 0, targets, ids, split='train', artifact_type='ku_leuven_materialized_known_train_validation')

    def test_member_aggregation_rejects_missing_and_mixed_labels(self):
        api = self.api()
        logits = np.repeat([[2., 0., 0.]], 32, axis=0)
        features = np.repeat([[1., 0., 0.]], 32, axis=0)
        result = api.aggregate_members(logits, features, np.zeros(32, dtype=int), ['a'] * 32, np.eye(3))
        self.assertEqual(result[0]['prediction'], 0)
        self.assertEqual(result[0]['cosine'], 1.)
        for labels, ids in [(np.zeros(32, int), ['a'] * 31 + ['b']), (np.arange(32) % 2, ['a'] * 32)]:
            with self.assertRaises(ValueError):
                api.aggregate_members(logits, features, labels, ids, np.eye(3))

    def test_checkpoint_rank_guard_and_earliest_tie(self):
        api = self.api()
        base = dict(clean_recording_macro_f1=1., minus5_recording_macro_f1=[.6, .8, .7], clean_window_loss=.1)
        np.testing.assert_allclose(api.checkpoint_rank(base), [.6, .7, 1., -.1])
        self.assertIsNone(api.checkpoint_rank(dict(base, clean_recording_macro_f1=.96)))
        chosen = api.select_checkpoint([dict(base, step=460), dict(base, step=230)])
        self.assertEqual(chosen['step'], 230)

    def test_group_selection_keeps_baseline_if_regression_gate_fails(self):
        api = self.api()
        groups = {k:dict(clean=1., zero_five=.8, multipath=.9, minus5_mean=.6, minus5_worst=.5, parameters=100, seeds_complete=True) for k in 'ABCD'}
        groups['B'].update(minus5_mean=.7, minus5_worst=.6, clean=.95)
        self.assertEqual(api.select_experiment(groups)['selected_experiment'], 'A')
        groups['C'].update(minus5_mean=.7, minus5_worst=.6, parameters=90)
        self.assertEqual(api.select_experiment(groups)['selected_experiment'], 'C')

    def test_missing_freeze_refuses_before_any_sealed_access(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ValueError, FileNotFoundError)):
                api.verify_freeze(Path(tmp) / 'freeze.json')

    def test_conditions_are_original_40_and_bootstrap_is_paired(self):
        api = self.api()
        conditions = api.conditions()
        self.assertEqual(len(conditions), 40)
        self.assertEqual(len({x['id'] for x in conditions}), 40)
        truth = np.array([0, 0, 1, 1, 2, 2])
        result = api.paired_bootstrap(truth, truth, truth, replicates=20)
        self.assertEqual(result['delta_macro_f1_ci95'], [0., 0.])

    def test_shard_resume_reuses_matching_result_and_rejects_tamper(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'cached_shard'), 'Resumable evaluation shards must exist')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shard.json'
            first = api.cached_shard(path, 'frozen-hash', lambda: {'value': 7})
            self.assertEqual(first, {'value': 7})
            second = api.cached_shard(path, 'frozen-hash', lambda: self.fail('completed shard recomputed'))
            self.assertEqual(second, first)
            with self.assertRaises(ValueError):
                api.cached_shard(path, 'different-freeze', lambda: {})
            path.write_text(path.read_text().replace('7', '8'))
            with self.assertRaises(ValueError):
                api.cached_shard(path, 'frozen-hash', lambda: {})

    def test_classification_intervals_are_member_stratified(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'classification_bootstrap'), 'Member-level metric intervals are required')
        rows = [dict(recording_id=str(i), label=i // 2, prediction=i // 2) for i in range(6)]
        result = api.classification_bootstrap(rows)
        self.assertEqual(result['macro_f1_ci95'], [1., 1.])
        self.assertEqual(result['accuracy_ci95'], [1., 1.])

    def test_freeze_checks_artifact_bytes_and_own_digest(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'model.pt'
            artifact.write_bytes(b'cpu-fixture')
            payload = dict(artifact_type='a800_v2_frozen_experiment', inference_precision='fp32', primary_score='cosine',
                           root_relative='.', runs=[{'run_id': 'fixture'}], bindings=[api._binding(root, artifact)])
            payload['content_sha256'] = api._digest(payload)
            path = root / 'freeze.json'
            path.write_text(json.dumps(payload))
            self.assertEqual(api.verify_freeze(path)['_root'], str(root.resolve()))
            artifact.write_bytes(b'changed-model')
            with self.assertRaises(ValueError):
                api.verify_freeze(path)
            artifact.write_bytes(b'cpu-fixture')
            payload['primary_score'] = 'msp'
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                api.verify_freeze(path)

    def test_completed_matrix_requires_committed_recovery_selection_and_current_data(self):
        """Small serialized states test integrity gates, not experiment performance."""
        api = self.api()
        from training import a800_runner as runner
        from training.a800_common import canonical_hash, source_identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'training').mkdir()
            (root / 'training/fixture.py').write_text('# integrity fixture\n')
            configs = [runner.make_run_config(e, s) for e in 'ABCD' for s in api.SEEDS]
            matrix = root / 'matrix.json'
            api.atomic_json(matrix, {'runs': configs})
            data = {'fixture': 'training/validation identity'}
            code = source_identity(root)
            for config in configs:
                directory = root / 'outputs/runs' / config['run_id']
                identity = dict(config_sha256=canonical_hash(config), code=code,
                                data={'train': data, 'validation': data}, precision=config['training']['precision'])
                metrics = dict(clean_recording_macro_f1=1., minus5_recording_macro_f1=[.6, .7, .8], clean_window_loss=.1)
                state = dict(schema_version='2.0', artifact_type='a800_training_checkpoint', kind='formal',
                             config=config, identity=identity, model_state={'weight': torch.tensor([1.])},
                             optimizer_state={}, scheduler_state={}, scaler=None, global_step=9200,
                             logical_epoch=40, epoch_step=0, draw_position=9200 * 128,
                             sampler={'epoch': 40, 'position': 0},
                             rng={key: None for key in ('python', 'numpy', 'torch_cpu', 'torch_cuda')},
                             best_rank=api.checkpoint_rank(metrics), best_step=230,
                             best_state={'weight': torch.tensor([1.])}, validation_pending=False,
                             timing={'elapsed_compute_seconds': 0., 'gpu_step_milliseconds': []},
                             history=[dict(event='validation', global_step=i, metrics=metrics) for i in range(230, 9201, 230)])
                runner._save_checkpoint(directory / 'last.pt', state, rotate=False)
                best = dict(state, global_step=230, logical_epoch=1, draw_position=230 * 128,
                            sampler={'epoch': 1, 'position': 0})
                runner._save_checkpoint(directory / 'best.pt', best, rotate=False)
                api.atomic_json(directory / 'run-status.json', dict(config_sha256=identity['config_sha256'],
                                global_step=9200, state='completed', checkpoint_sha256=api.sha256_file(directory / 'last.pt'),
                                best_checkpoint_sha256=api.sha256_file(directory / 'best.pt')))
                api.atomic_json(directory / 'data-access-audit.json', dict(sealed_access_count=0, test_accessed=False,
                                unknown_accessed=False, splits=['train', 'validation'], inputs=identity['data'],
                                config_sha256=identity['config_sha256']))
            directory = root / 'outputs/runs' / configs[0]['run_id']
            original = runner.load_checkpoint(directory / 'last.pt')
            original_status = api.read_json(directory / 'run-status.json')
            def replace_state(state):
                runner._save_checkpoint(directory / 'last.pt', state, rotate=False)
                api.atomic_json(directory / 'run-status.json', dict(original_status, checkpoint_sha256=api.sha256_file(directory / 'last.pt')))
            with patch.object(runner, '_datasets', return_value=(Mock(), Mock())), patch.object(runner, '_dataset_identity', return_value=data):
                self.assertEqual(len(api._completed_runs(root, matrix)[0]), 12)
                mutations = [lambda s: s.update(validation_pending=True),
                             lambda s: s['history'].pop(),
                             lambda s: s['identity'].update(data={'train': {'changed': True}, 'validation': data}),
                             lambda s: s.update(best_step=460),
                             lambda s: s.update(best_state={'weight': torch.tensor([2.])})]
                for mutation in mutations:
                    changed = copy.deepcopy(original)
                    mutation(changed)
                    replace_state(changed)
                    with self.assertRaises(ValueError):
                        api._completed_runs(root, matrix)
                replace_state(original)
                receipt = directory / 'last.pt.sha256.json'
                receipt.write_text('{}')
                with self.assertRaises(ValueError):
                    api._completed_runs(root, matrix)

    def test_finalization_requires_allocation_before_cuda_or_sealed_reads(self):
        api = self.api()
        with patch.object(torch.cuda, 'set_device', side_effect=AssertionError('CUDA started')):
            with self.assertRaises(ValueError):
                api.start_finalization(None)

    def test_freeze_environment_requires_real_lock_and_verified_server_receipts(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                api._environment_paths(root)
            api.atomic_json(root / 'environment/linux-resolve.json', {'install': [
                {'metadata': {'name': 'fixture-only-package', 'version': '1.0'}}]})
            api.atomic_json(root / 'environment/candidate-status.json', {'fixture': True})
            (root / 'environment/requirements-linux-lock.txt').write_text('fixture-only-package==1.0\n')
            (root / 'environment/requirements-training.txt').write_text('fixture-only-package==1.0\n')
            snapshot = root / 'outputs/evidence/environment/installed-environment.json'
            api.atomic_json(snapshot, {'linux_environment_verified': False, 'packages': {'fixture-only-package': '1.0'}})
            (snapshot.parent / 'pip-freeze.txt').write_text('fixture-only-package==1.0\n')
            with self.assertRaises(ValueError):
                api._environment_paths(root)
            api.atomic_json(snapshot, {'linux_environment_verified': True, 'packages': {'fixture-only-package': '1.0'}})
            paths = api._environment_paths(root)
            self.assertEqual(len(paths), 6)
            payload = dict(schema_version='2.0', artifact_type='a800_v2_frozen_experiment', inference_precision='fp32',
                           primary_score='cosine', root_relative='.', runs=[{'run_id': 'fixture'}],
                           environment_files=sorted(p.relative_to(root).as_posix() for p in paths),
                           bindings=[api._binding(root, p) for p in paths])
            payload['content_sha256'] = api._digest(payload)
            freeze = root / 'freeze.json'
            api.atomic_json(freeze, payload)
            api.verify_freeze(freeze)
            (snapshot.parent / 'pip-freeze.txt').write_text('fixture-only-package==2.0\n')
            with self.assertRaises(ValueError):
                api.verify_freeze(freeze)

    def test_historical_references_are_registered_separately_from_selection(self):
        api = self.api()
        root = Path(__file__).resolve().parents[1]
        runs = api._historical_runs(root)
        self.assertEqual([r['config']['seed'] for r in runs], api.SEEDS)
        self.assertTrue(all(r['config']['experiment'] == 'V4' and not r['config']['budget_comparable_to_a800_v2'] for r in runs))
        self.assertTrue(all(not g['seeds_complete'] for g in api._group_summary([{'config': r['config']} for r in runs]).values()))


if __name__ == '__main__':
    unittest.main()
