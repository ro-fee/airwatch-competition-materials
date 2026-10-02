"""Small CPU fixtures for source binding and sealed-reader resource boundaries."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import h5py
import numpy as np

from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_preprocessing import PREPROCESSING_ID
from airwatch.data.ku_leuven_training_v2 import A800Dataset
from training import a800_data


class DenseIntegrityTests(unittest.TestCase):
    def _fixture(self, root, change=None):
        split = root / 'train'
        split.mkdir()
        waveform = np.stack((np.sin(np.arange(4096)), np.cos(np.arange(4096))))
        data = np.broadcast_to(waveform, (256, 2, 4096)).astype(np.float32).copy()
        labels = np.zeros(256, dtype=np.int64)
        rows = [dict(data_index=i, window_id=f'recording:w{i}', recording_id='recording',
                     archive_id='frysky-v1', label_index=0, member_index=0,
                     member_path='Frysky/Frysky_0.mat', member_sha256='a' * 64,
                     split='train', start_sample=i * 4096,
                     end_sample_exclusive=(i + 1) * 4096,
                     preprocessing_id=PREPROCESSING_ID) for i in range(256)]
        if change:
            change(rows, labels)
        np.save(split / 'data.npy', data)
        np.save(split / 'labels.npy', labels)
        with (split / 'windows.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        artifacts = {name: dict(path=f'train/{name}', size_bytes=(split / name).stat().st_size,
                                sha256=sha256_file(split / name))
                     for name in ('data.npy', 'labels.npy', 'windows.csv')}
        metadata = dict(schema_version='2.0', artifact_type='ku_leuven_a800_v2',
                        preprocessing_id=PREPROCESSING_ID, shape_per_window=[2, 4096],
                        dtype='float32', split_counts={'train': 256},
                        source_split_assignments={'sha256': 'b' * 64},
                        source_window_plans=[{'sha256': 'c' * 64}],
                        artifacts={'train': artifacts})
        (root / 'dataset-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    def test_consistent_dense_recording_loads_and_returns_independent_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            with A800Dataset(root) as dataset:
                values, label, index = dataset[0]
                values.fill_(0)
                self.assertEqual((label, index, len(dataset)), (0, 0, 256))
                self.assertGreater(dataset[0][0].abs().sum().item(), 0)

    def test_same_recording_cannot_have_mixed_label_or_raw_member_identity(self):
        mutations = [
            lambda rows, labels: (rows[1].update(label_index=1), labels.__setitem__(1, 1)),
            lambda rows, labels: rows[1].update(member_path='different.mat'),
            lambda rows, labels: rows[1].update(member_sha256='b' * 64),
            lambda rows, labels: rows[1].update(archive_id='mini2-rc-v1'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._fixture(root, mutation)
                with self.assertRaises(ValueError):
                    with A800Dataset(root):
                        pass

    def test_rejects_nonhex_member_checksum_even_when_csv_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, lambda rows, _: [r.update(member_sha256='z' * 64) for r in rows])
            with self.assertRaises(ValueError):
                with A800Dataset(root):
                    pass


class KnownInputBarrierTests(unittest.TestCase):
    def _fixture(self, root):
        project = Path(__file__).resolve().parents[1]
        candidates = [project / 'manifests', project / 'datasets/uav/ku_leuven_drone_rf']
        source = next((p for p in candidates if (p / 'known-split-v1/bundle-index.json').is_file()), None)
        self.assertIsNotNone(source, 'Frozen known manifests must ship with the CPU integrity tests')
        manifests = root / 'manifests'
        manifests.mkdir()
        for folder in ['known-split-v1', *(entry[2] for entry in a800_data.KNOWN)]:
            shutil.copytree(source / folder, manifests / folder)
        old = root / 'v1'
        old.mkdir()
        split = manifests / 'known-split-v1/known-split-assignments.csv'
        meta = {'source_split_assignments': {'sha256': sha256_file(split)},
                'source_window_plans': [{'sha256': sha256_file(manifests / entry[2] / 'window-plan.csv')}
                                        for entry in a800_data.KNOWN]}
        (old / 'dataset-metadata.json').write_text(json.dumps(meta), encoding='utf-8')
        raw = root / 'raw'
        raw.mkdir()
        return manifests, raw, old

    def _barrier(self):
        barrier = getattr(a800_data, 'verify_known_inputs', None)
        self.assertTrue(callable(barrier), 'verify_known_inputs must precede all raw materialization')
        return barrier

    def test_relabelled_split_is_rejected_before_any_archive_is_opened(self):
        barrier = self._barrier()
        with tempfile.TemporaryDirectory() as temporary:
            manifests, raw, old = self._fixture(Path(temporary))
            path = manifests / 'known-split-v1/known-split-assignments.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            train = next(row for row in rows if row['split'] == 'train')
            test = next(row for row in rows if row['split'] == 'test')
            train['split'], test['split'] = test['split'], train['split']
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch('zipfile.ZipFile', side_effect=AssertionError('raw read before source audit')):
                with self.assertRaisesRegex(ValueError, '(?i)split|frozen|划分|identity|hash'):
                    barrier(manifests, raw, old)

    def test_missing_indexed_plan_is_rejected_before_any_archive_is_opened(self):
        barrier = self._barrier()
        with tempfile.TemporaryDirectory() as temporary:
            manifests, raw, old = self._fixture(Path(temporary))
            (manifests / a800_data.KNOWN[0][2] / 'window-plan.csv').unlink()
            with patch('zipfile.ZipFile', side_effect=AssertionError('raw read before source audit')):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    barrier(manifests, raw, old)


class SealedReaderIntegrityTests(unittest.TestCase):
    def _fixture(self, root, *, starts=None):
        values = np.zeros(32 * 4096, dtype=[('real', '<f8'), ('imag', '<f8')])
        values['real'] = np.sin(np.arange(len(values)) / 13)
        values['imag'] = np.cos(np.arange(len(values)) / 13)
        stream = io.BytesIO()
        with h5py.File(stream, 'w') as handle:
            data = handle.create_dataset('uhd_samps', shape=(1, 10000000), dtype=values.dtype,
                                         chunks=(1, 4096), compression='gzip')
            data[0, :len(values)] = values
            data.attrs['MATLAB_class'] = np.bytes_('double')
        payload = stream.getvalue()
        with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
            archive.writestr('source/member_0.mat', payload)
        return dict(recording_id='recording', archive_id='frysky-v1', label_index=0,
                    archive_path='source.zip', member_path='source/member_0.mat',
                    member_size_bytes=len(payload), member_sha256=hashlib.sha256(payload).hexdigest(),
                    sample_count=10000000, split='test', preprocessing_id=PREPROCESSING_ID,
                    sample_rate_hz=100000000,
                    window_starts=list(range(0, 32 * 4096, 4096)) if starts is None else starts)

    def test_checksum_bound_member_returns_only_32_normalized_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = self._fixture(root)
            result = a800_data.load_sealed_member(root, member)
            self.assertEqual(result.shape, (32, 2, 4096))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result.mean(axis=2), 0, atol=1e-6)
            np.testing.assert_allclose((result ** 2).sum(axis=1).mean(axis=1), 1, atol=1e-6)
            wrong = copy.deepcopy(member)
            wrong['member_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                a800_data.load_sealed_member(root, wrong)

    def test_32_distinct_but_overlapping_windows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = self._fixture(root, starts=list(range(32)))
            with self.assertRaises(ValueError):
                a800_data.load_sealed_member(root, member)

    def test_claimed_member_size_must_match_observed_zip_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = self._fixture(root)
            member['member_size_bytes'] += 1
            with self.assertRaises(ValueError):
                a800_data.load_sealed_member(root, member)

    def test_claimed_sample_count_must_match_observed_hdf5_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = self._fixture(root)
            member['sample_count'] += 1
            with self.assertRaises(ValueError):
                a800_data.load_sealed_member(root, member)

    def test_oversized_zip_member_is_rejected_from_header_before_payload_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / 'source.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('member.mat', b'fixture')
            payload = bytearray(archive_path.read_bytes())
            central = payload.index(b'PK\x01\x02')
            declared_size = 64 * 1024 * 1024 + 1
            struct.pack_into('<L', payload, central + 24, declared_size)
            struct.pack_into('<L', payload, 22, declared_size)
            archive_path.write_bytes(payload)
            member = dict(archive_path='source.zip', member_path='member.mat',
                          member_size_bytes=declared_size, member_sha256='a' * 64,
                          sample_count=10000000, sample_rate_hz=100000000,
                          split='test', preprocessing_id=PREPROCESSING_ID,
                          window_starts=list(range(0, 32 * 4096, 4096)))
            # The actual central directory declares the oversized member. Only
            # payload I/O is trapped, so no large allocation is needed to prove
            # the guard runs before decompression or checksum streaming.
            with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('oversized payload read')):
                with self.assertRaises(ValueError):
                    a800_data.load_sealed_member(root, member)


if __name__ == '__main__':
    unittest.main()
