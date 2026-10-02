"""Runtime contract CPU fixtures are independent of formal model evidence."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class ContractTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('airwatch.inference.uav_a800_contract'),
                             'Reloadable A800 runtime contract is required')
        from airwatch.inference import uav_a800_contract
        return uav_a800_contract

    def test_member_scores_use_mean_probability_and_normalized_member_features(self):
        api = self.api()
        logits = np.tile([[2., 0., 0.]], (32, 1))
        features = np.tile([[1., 0., 0.]], (32, 1))
        result = api.member_scores(logits, features, np.eye(3))
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['cosine'], 1.)
        self.assertAlmostEqual(result['msp'], .7869860421615985)
        with self.assertRaises(ValueError):
            api.member_scores(logits[:31], features[:31], np.eye(3))
        with self.assertRaises(ValueError):
            api.member_scores(logits, features * 0, np.eye(3))

    def test_export_reloads_predicts_and_rejects_weights_and_metadata_tamper(self):
        api = self.api()
        self.assertIsNotNone(importlib.util.find_spec('training.a800_delivery'), 'Runtime exporter is required')
        from training.a800_delivery import write_runtime_contract
        from airwatch.models.uav_multiscale_tcn import build_a800_model
        torch.set_num_threads(1)
        model = build_a800_model('tcn').eval()
        prototypes = np.eye(3, 256)
        thresholds = {s: {'threshold': -2., 'comparison': 'score>=threshold', 'direction': 'higher_is_known'} for s in ('cosine', 'msp', 'energy')}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'release'
            path = write_runtime_contract(model, 'tcn', prototypes, thresholds, output,
                                          source_root=Path(__file__).resolve().parents[1], model_version='cpu-test-fixture')
            contract = api.load_contract(path, device='cpu')
            values = np.random.default_rng(8).normal(size=(32, 2, 4096)).astype(np.float32)
            result = api.predict_member(contract, values, sample_rate_hz=100000000)
            self.assertEqual(result['window_count'], 32)
            self.assertEqual(result['decision'], 'known')
            self.assertGreater(result['elapsed_seconds'], 0)
            self.assertEqual(result['model_version'], 'cpu-test-fixture')
            # Reload the shipped source in an independent process with no project
            # directory on PYTHONPATH; this tests the actual runtime dependency closure.
            environment = dict(os.environ, PYTHONPATH=str(output / 'source'))
            script = ('import sys,torch; torch.set_num_threads(1); '
                      'from airwatch.inference.uav_a800_contract import load_contract; '
                      'c=load_contract(sys.argv[1]); print(c["model_version"])')
            completed = subprocess.run([sys.executable, '-c', script, str(path)],
                                       cwd=tmp, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), 'cpu-test-fixture')
            with self.assertRaises(ValueError):
                api.predict_member(contract, values, sample_rate_hz=20000000)
            with self.assertRaises(ValueError):
                api.predict_member(contract, values[:31], sample_rate_hz=100000000)
            weights = output / 'model.pt'
            original = weights.read_bytes()
            with weights.open('ab') as handle:
                handle.write(b'tamper')
            with self.assertRaises(ValueError):
                api.load_contract(path, device='cpu')
            weights.write_bytes(original)
            (output / 'preprocessing.json').write_text('{}')
            with self.assertRaises(ValueError):
                api.load_contract(path, device='cpu')

    def test_result_packaging_refuses_unfinished_release_without_gpu_work(self):
        self.api()
        from training import a800_delivery
        self.assertTrue(hasattr(a800_delivery, 'package_results'), 'Verified CPU result packaging is required')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises((ValueError, FileNotFoundError)):
                a800_delivery.package_results(root, root / 'results.tar.gz')
            self.assertFalse((root / 'results.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
