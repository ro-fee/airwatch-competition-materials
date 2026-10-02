"""CPU-only result packaging fixtures; never produce formal experiment evidence."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from training import a800_delivery as delivery
from training import a800_evaluation as evaluation


class DeliveryTests(unittest.TestCase):
    def fixture(self, root):
        evidence = root / 'outputs/evidence'
        release = root / 'outputs/release'
        release.mkdir(parents=True)
        # Opaque stand-in files exercise transport checks, not model quality.
        model = release / 'fixture.txt'
        model.write_text('synthetic packaging fixture', encoding='utf-8')
        freeze = evidence / 'freeze.json'
        payload = dict(artifact_type='a800_v2_frozen_experiment', inference_precision='fp32',
                       primary_score='cosine', root_relative='../..', runs=[{'run_id': 'fixture'}],
                       bindings=[evaluation._binding(root, model)])
        payload['content_sha256'] = evaluation._digest(payload)
        evaluation.atomic_json(freeze, payload)
        identity = evaluation.sha256_file(freeze)
        evaluation.atomic_json(release / 'release-completed.json', dict(status='completed', freeze_sha256=identity,
                               files=[dict(path=model.name, sha256=evaluation.sha256_file(model))]))
        (release / 'SHA256SUMS').write_text('\n'.join(f'{evaluation.sha256_file(p)}  {p.name}'
                                                     for p in sorted(release.iterdir())) + '\n', encoding='utf-8')
        test = evidence / 'test'
        evaluation.atomic_json(test / 'metrics.json', {'fixture': True})
        evaluation.atomic_json(test / 'evaluation-completed.json', dict(status='completed', freeze_sha256=identity,
                               artifacts=[dict(path='metrics.json', sha256=evaluation.sha256_file(test / 'metrics.json'))]))
        benchmark = evidence / 'benchmark'
        evaluation.atomic_json(benchmark / 'raw.json', {'fixture': True})
        evaluation.atomic_json(benchmark / 'benchmark.json', dict(freeze_sha256=identity,
                               raw_files=[dict(path='raw.json', sha256=evaluation.sha256_file(benchmark / 'raw.json'))]))
        return release, test, benchmark

    def test_package_cli_archives_verified_results_without_gpu_or_inference(self):
        with tempfile.TemporaryDirectory(prefix='CPU package ') as tmp:
            root = Path(tmp)
            self.fixture(root)
            output = root / 'portable results.tar.gz'
            with patch.object(delivery.torch.cuda, 'is_available', side_effect=AssertionError('GPU access')):
                self.assertEqual(delivery.main(['package', '--root', str(root), '--output', str(output)]), 0)
            with tarfile.open(output, 'r:gz') as archive:
                self.assertIn('outputs/release/fixture.txt', archive.getnames())
                self.assertIn('outputs/evidence/freeze.json', archive.getnames())
                self.assertIn('outputs/evidence/benchmark/raw.json', archive.getnames())
            receipt = json.loads(output.with_name(output.name + '.sha256.json').read_text())
            self.assertEqual(receipt['sha256'], evaluation.sha256_file(output))
            with self.assertRaises(FileExistsError):
                delivery.package_results(root, output)

    def test_package_refuses_interrupted_marker_or_changed_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release, _, benchmark = self.fixture(root)
            marker = release / 'release-completed.json'
            original = evaluation.read_json(marker)
            evaluation.atomic_json(marker, dict(original, status='interrupted'))
            with self.assertRaises(ValueError):
                delivery.package_results(root, root / 'bad.tar.gz')
            evaluation.atomic_json(marker, original)
            evaluation.atomic_json(benchmark / 'raw.json', {'changed': True})
            with self.assertRaises(ValueError):
                delivery.package_results(root, root / 'bad.tar.gz')
            self.assertFalse((root / 'bad.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
