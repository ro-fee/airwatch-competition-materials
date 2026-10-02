import json
import tempfile
import unittest
from pathlib import Path
from airwatch.data.data_provenance import audit_manifest, inventory_files, sha256_file


class DataProvenanceTests(unittest.TestCase):
    def test_inventory_is_sorted_and_hashes_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'b.dat').write_bytes(b'b'); (root / 'a.npy').write_bytes(b'a')
            records = inventory_files(root, suffixes=('.dat', '.npy'))
            self.assertEqual([r.path for r in records], ['a.npy', 'b.dat'])
            self.assertEqual(records[0].sha256, sha256_file(root / 'a.npy'))

    def test_project_manifest_releases_verified_dronerf_training_data(self):
        manifest = json.loads(Path('docs/uav_data_sources.json').read_text(encoding='utf-8'))
        report = audit_manifest(manifest, project_root=Path('.'))
        self.assertTrue(report['ok'])
        self.assertTrue(report['ready_for_training'])
        self.assertEqual(report['primary_sources'], ['DroneRF'])
        self.assertFalse(any(item.get('severity') == 'error' for item in report['findings']))
        self.assertTrue(any('许可证尚未核实' in item.get('message', '') for item in report['findings']))
        self.assertTrue(any(item.get('file_count') is not None for item in report['findings']))

    def test_audit_rejects_changed_local_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'raw'
            raw.mkdir()
            training = root / 'training'
            training.mkdir()
            (training / 'verification.json').write_text(json.dumps({
                'ok': True, 'findings': [], 'dataset': 'ready', 'dataset_version': '1',
                'verified_recording_files': 1, 'verified_window_rows': 1,
            }), encoding='utf-8')
            sample = raw / 'sample.csv'
            sample.write_text('1\n2\n', encoding='utf-8')
            manifest = {
                'sources': [{
                    'name': 'ready', 'role': 'primary', 'status': 'downloaded',
                    'source_url': 'https://example.invalid/data', 'license': 'CC BY 4.0',
                    'license_verified': True, 'version': '1', 'download_date': '2026-09-08',
                    'checksum': 'abc', 'local_root': 'raw',
                    'training_data_root': 'training',
                    'training_verification_path': 'training/verification.json',
                }],
                'local_inventory': {
                    'name': 'local', 'root': 'raw', 'suffixes': ['.csv'],
                    'files': [{'path': 'sample.csv', 'size_bytes': 1, 'sha256': 'bad'}],
                },
            }
            report = audit_manifest(manifest, project_root=root)
            self.assertFalse(report['ready_for_training'])
            self.assertTrue(any('SHA-256' in item.get('message', '') for item in report['findings']))

    def test_audit_accepts_verified_primary_and_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'raw'
            raw.mkdir()
            training = root / 'training'
            training.mkdir()
            (training / 'verification.json').write_text(json.dumps({
                'ok': True, 'findings': [], 'dataset': 'ready', 'dataset_version': '1',
                'verified_recording_files': 1, 'verified_window_rows': 1,
            }), encoding='utf-8')
            sample = raw / 'sample.csv'
            sample.write_text('1\n2\n', encoding='utf-8')
            manifest = {
                'sources': [{
                    'name': 'ready', 'role': 'primary', 'status': 'downloaded',
                    'source_url': 'https://example.invalid/data', 'license': 'CC BY 4.0',
                    'license_verified': True, 'version': '1', 'download_date': '2026-09-08',
                    'checksum': 'abc', 'local_root': 'raw',
                    'training_data_root': 'training',
                    'training_verification_path': 'training/verification.json',
                }],
                'local_inventory': {
                    'name': 'local', 'root': 'raw', 'suffixes': ['.csv'],
                    'files': [{
                        'path': 'sample.csv', 'size_bytes': sample.stat().st_size,
                        'sha256': sha256_file(sample),
                    }],
                },
            }
            report = audit_manifest(manifest, project_root=root)
            self.assertTrue(report['ok'])
            self.assertTrue(report['ready_for_training'])

    def test_audit_verifies_external_archive_without_promoting_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / 'external.7z'
            archive.write_bytes(b'verified external archive')
            manifest = {
                'sources': [{
                    'name': 'external', 'role': 'external_candidate',
                    'status': 'downloaded but blocked',
                    'source_url': 'https://example.invalid/data',
                    'license': 'CC BY 4.0', 'license_verified': True,
                    'version': '1', 'archive_local_path': 'external.7z',
                    'archive_bytes': archive.stat().st_size,
                    'archive_sha256': sha256_file(archive),
                    'training_eligible': False,
                    'training_blocker': 'password required',
                }],
            }
            report = audit_manifest(manifest, project_root=root, verify_archives=True)
            self.assertFalse(report['ready_for_training'])
            self.assertEqual(report['primary_sources'], [])
            self.assertTrue(any(
                item.get('source') == 'external'
                and item.get('severity') == 'info'
                and 'SHA-256 已复核' in item.get('message', '')
                for item in report['findings']
            ))
            self.assertTrue(any(
                item.get('source') == 'external'
                and item.get('message') == 'password required'
                for item in report['findings']
            ))


if __name__ == '__main__':
    unittest.main()
