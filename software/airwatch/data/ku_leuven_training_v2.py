"""Immutable dense training plans and hash-bound A800 data access."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_provenance import sha256_file
from .ku_leuven_dataset import KULeuvenWindowSample, KU_LEUVEN_KNOWN_LABELS
from .ku_leuven_preprocessing import PREPROCESSING_ID, INPUT_WINDOW_SAMPLES

def dense_offsets(recording_id: str, sample_count: int, original: list[int]) -> list[int]:
    """Keep V1's 32 windows; sample one aligned window from each of 224 strata."""
    length = INPUT_WINDOW_SAMPLES
    old = sorted(int(x) for x in original)
    if len(old) != 32 or old[0] < 0 or old[-1] + length > sample_count:
        raise ValueError('Expected 32 bounded original windows')
    if any(b < a + length for a, b in zip(old, old[1:])):
        raise ValueError('Original windows overlap')
    candidates = np.arange(0, sample_count - length + 1, length, dtype=np.int64)
    keep = np.ones(len(candidates), dtype=bool)
    for start in old:
        keep &= (candidates + length <= start) | (candidates >= start + length)
    candidates = candidates[keep]
    if len(candidates) < 224:
        raise ValueError('Insufficient nonoverlapping dense candidates')
    seed = int.from_bytes(hashlib.sha256(('dense-v2|20260929|' + recording_id).encode()).digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = [int(group[rng.integers(len(group))]) for group in np.array_split(candidates, 224)]
    return sorted(old + selected)

class A800Dataset(Dataset):
    """V2-only array reader; does not widen the historical V1 reader contract."""
    def __init__(self, root, split='train', verify_hashes=True):
        if split not in {'train', 'validation'}:
            raise ValueError('Training reader refuses test/unknown/guard/external data')
        self.root, self.split = Path(root), split
        self.metadata = json.loads((self.root / 'dataset-metadata.json').read_text(encoding='utf-8'))
        if self.metadata.get('artifact_type') != 'ku_leuven_a800_v2' or self.metadata.get('preprocessing_id') != PREPROCESSING_ID:
            raise ValueError('Incompatible dense data contract')
        self._data = self._labels = None
        specs = self.metadata['artifacts'][split]
        for name in ('data.npy', 'labels.npy', 'windows.csv'):
            path = self.root / split / name
            if specs[name]['size_bytes'] != path.stat().st_size or (verify_hashes and sha256_file(path) != specs[name]['sha256']):
                raise ValueError(f'Data integrity failure: {split}/{name}')
        self._data = np.load(self.root / split / 'data.npy', mmap_mode='r', allow_pickle=False)
        self._labels = np.load(self.root / split / 'labels.npy', mmap_mode='r', allow_pickle=False)
        count = self.metadata['split_counts'][split]
        if self._data.shape != (count, 2, 4096) or self._data.dtype != np.float32 or self._labels.shape != (count,) or self._labels.dtype != np.int64:
            raise ValueError('Invalid dense array shape or dtype')
        for offset in range(0, count, 256):
            if not np.isfinite(self._data[offset:offset+256]).all():
                raise ValueError('Nonfinite dense input')
        with (self.root / split / 'windows.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count:
            raise ValueError('Dense trace row count mismatch')
        integer_fields = {'data_index', 'label_index', 'member_index', 'start_sample', 'end_sample_exclusive'}
        self.samples = tuple(KULeuvenWindowSample(**{k: int(v) if k in integer_fields else v for k,v in row.items()}) for row in rows)
        seen = set()
        by_recording = {}
        for i, sample in enumerate(self.samples):
            if sample.data_index != i or sample.split != split or sample.preprocessing_id != PREPROCESSING_ID or sample.label_index != int(self._labels[i]) or not 0 <= sample.label_index <= 2 or sample.start_sample < 0 or sample.end_sample_exclusive - sample.start_sample != 4096 or not re.fullmatch('[0-9a-f]{64}',sample.member_sha256) or sample.window_id in seen:
                raise ValueError('Dense provenance mismatch')
            seen.add(sample.window_id)
            by_recording.setdefault(sample.recording_id, []).append(sample)
        expected = 256 if split == 'train' else 32
        for members in by_recording.values():
            if len({(x.archive_id,x.member_path,x.member_index,x.member_sha256,x.label_index) for x in members})!=1:
                raise ValueError('Mixed physical member identity in one recording')
            if len(members) != expected:
                raise ValueError('Wrong windows per recording')
            members.sort(key=lambda x: x.start_sample)
            if any(b.start_sample < a.end_sample_exclusive for a,b in zip(members,members[1:])):
                raise ValueError('Dense windows overlap')
        self.recording_ids = frozenset(by_recording)
        self.label_map = dict(KU_LEUVEN_KNOWN_LABELS)
        self.num_classes, self.channels, self.window_size = 3, 2, 4096
        self.data_identity = {'metadata_sha256': sha256_file(self.root / 'dataset-metadata.json'), 'artifact_type': 'ku_leuven_a800_v2', 'split': split, 'artifacts': specs, 'preprocessing_id': PREPROCESSING_ID}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int,np.integer)):
            raise TypeError('index must be integer')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return torch.from_numpy(np.array(self._data[index], copy=True)), int(self._labels[index]), int(index)

    def sample_metadata(self, index):
        return self.samples[index]

    def close(self):
        for name in ('_data','_labels'):
            mmap = getattr(getattr(self, name, None), '_mmap', None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
