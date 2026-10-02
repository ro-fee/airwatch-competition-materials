import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_dataset import (
    KULeuvenDatasetError,
    KULeuvenMaterializedDataset,
)


class KULeuvenDatasetTests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path) -> None:
        artifacts = {}
        split_counts = {"train": 2, "validation": 1}
        for split, count in split_counts.items():
            split_dir = root / split
            split_dir.mkdir(parents=True)
            data = np.arange(count * 2 * 4096, dtype=np.float32).reshape(count, 2, 4096)
            labels = np.arange(count, dtype=np.int64)
            np.save(split_dir / "data.npy", data)
            np.save(split_dir / "labels.npy", labels)
            rows = []
            for index in range(count):
                rows.append({
                    "data_index": index,
                    "window_id": f"{split}-r{index}:w000",
                    "recording_id": f"{split}-r{index}",
                    "archive_id": "frysky-v1",
                    "label_index": int(labels[index]),
                    "member_index": index,
                    "member_path": f"member-{index}.mat",
                    "member_sha256": f"{index + 1:064x}",
                    "split": split,
                    "start_sample": 0,
                    "end_sample_exclusive": 4096,
                    "preprocessing_id": "ku-leuven-iq-dc-rms-v1",
                })
            with (split_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            artifacts[split] = {
                name: {
                    "size_bytes": (split_dir / name).stat().st_size,
                    "sha256": sha256_file(split_dir / name),
                }
                for name in ("data.npy", "labels.npy", "windows.csv")
            }
        metadata = {
            "artifact_type": "ku_leuven_materialized_known_train_validation",
            "preprocessing_id": "ku-leuven-iq-dc-rms-v1",
            "split_counts": split_counts,
            "source_split_assignments": {"sha256": "a" * 64},
            "source_window_plans": [{"sha256": "b" * 64}],
            "artifacts": artifacts,
        }
        (root / "dataset-metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_loads_hash_bound_arrays_and_returns_independent_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            dataset = KULeuvenMaterializedDataset(root, split="train")
            values, label, index = dataset[0]
            self.assertEqual(tuple(values.shape), (2, 4096))
            self.assertEqual((label, index), (0, 0))
            values[0, 0] = -1
            self.assertEqual(float(dataset[0][0][0, 0]), 0)
            self.assertEqual(len(dataset.recording_ids), 2)
            self.assertEqual(dataset.sample_metadata(1).recording_id, "train-r1")
            dataset.close()

    def test_train_and_validation_share_identity_but_not_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            train = KULeuvenMaterializedDataset(root, split="train")
            validation = KULeuvenMaterializedDataset(root, split="validation")
            self.assertEqual(train.data_identity, validation.data_identity)
            self.assertFalse(train.recording_ids & validation.recording_ids)
            train.close()
            validation.close()

    def test_rejects_test_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with self.assertRaises(KULeuvenDatasetError):
                KULeuvenMaterializedDataset(root, split="test")
            with (root / "train" / "labels.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(KULeuvenDatasetError):
                KULeuvenMaterializedDataset(root, split="train")


if __name__ == "__main__":
    unittest.main()
