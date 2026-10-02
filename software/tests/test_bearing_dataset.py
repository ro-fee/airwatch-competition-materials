import csv
import json
import unittest
from pathlib import Path

import numpy as np

from airwatch.data import CWRUBearingDataset


ROOT = Path(__file__).resolve().parents[1]
CWRU_ROOT = ROOT / "datasets" / "bearing" / "cwru"
MANIFEST = CWRU_ROOT / "split-manifest.csv"
LABEL_MAP = CWRU_ROOT / "label-map.json"


class TestCWRUBearingDataset(unittest.TestCase):
    def test_expands_audited_manifest_to_expected_windows(self):
        expected = {"train": 1422, "validation": 828, "test": 831}
        for split, count in expected.items():
            dataset = CWRUBearingDataset(
                MANIFEST,
                split=split,
                label_map_path=LABEL_MAP,
                project_root=ROOT,
            )
            self.assertEqual(len(dataset), count)
            self.assertEqual(dataset.window_size, 1024)
            self.assertEqual(dataset.step, 1024)
            self.assertEqual(dataset.num_classes, 4)

    def test_item_shape_and_label_are_from_manifest(self):
        dataset = CWRUBearingDataset(
            MANIFEST,
            split="validation",
            label_map_path=LABEL_MAP,
            project_root=ROOT,
        )
        signal, label = dataset[0]
        self.assertEqual(tuple(signal.shape), (1, 1024))
        self.assertEqual(signal.dtype.name if hasattr(signal.dtype, "name") else str(signal.dtype), "torch.float32")
        self.assertIn(label, range(4))
        metadata = dataset.sample_metadata(0)
        self.assertEqual(metadata.label_index, label)
        self.assertEqual(metadata.split, "validation")

    def test_window_zscore_normalizes_each_returned_window(self):
        dataset = CWRUBearingDataset(
            MANIFEST,
            split="validation",
            label_map_path=LABEL_MAP,
            normalization="window_zscore",
            project_root=ROOT,
        )

        signal, _ = dataset[0]
        values = signal.numpy()[0]
        self.assertAlmostEqual(float(values.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(values.std()), 1.0, places=5)

    def test_normal_2_uses_explicit_audited_channel(self):
        dataset = CWRUBearingDataset(
            MANIFEST,
            split="validation",
            label_map_path=LABEL_MAP,
            project_root=ROOT,
        )
        selected = [sample for sample in (dataset.sample_metadata(i) for i in range(len(dataset))) if sample.original_filename == "Normal_2.mat"]
        self.assertTrue(selected)
        self.assertEqual({sample.sensor_key for sample in selected}, {"X099_DE_time"})

    def test_source_recordings_and_split_groups_do_not_cross_splits(self):
        datasets = {
            split: CWRUBearingDataset(
                MANIFEST,
                split=split,
                label_map_path=LABEL_MAP,
                project_root=ROOT,
            )
            for split in ("train", "validation", "test")
        }
        source_sets = {
            split: {str(path) for path in dataset.source_files}
            for split, dataset in datasets.items()
        }
        self.assertFalse(source_sets["train"] & source_sets["validation"])
        self.assertFalse(source_sets["train"] & source_sets["test"])
        self.assertFalse(source_sets["validation"] & source_sets["test"])

    def test_dataset_does_not_create_processed_directory(self):
        self.assertFalse((CWRU_ROOT / "processed").exists())
        CWRUBearingDataset(
            MANIFEST,
            split="test",
            label_map_path=LABEL_MAP,
            project_root=ROOT,
        )
        self.assertFalse((CWRU_ROOT / "processed").exists())


if __name__ == "__main__":
    unittest.main()
