import unittest
from pathlib import Path

import numpy as np

from airwatch.data import DRONE_TYPE_LABELS, DroneRFWindowDataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "uav" / "dronerf" / "processed" / (
    "dronerf-two-band-raw-windows-v1-w4096-n32-seed20260908"
)


class DroneRFDatasetTests(unittest.TestCase):
    def test_fixed_split_counts_and_recording_isolation(self):
        datasets = {
            split: DroneRFWindowDataset(DATASET_ROOT, split=split)
            for split in ("train", "validation", "test")
        }
        self.assertEqual({key: len(value) for key, value in datasets.items()}, {
            "train": 5152, "validation": 1056, "test": 1056
        })
        self.assertFalse(datasets["train"].recording_ids & datasets["validation"].recording_ids)
        self.assertFalse(datasets["train"].recording_ids & datasets["test"].recording_ids)
        self.assertFalse(datasets["validation"].recording_ids & datasets["test"].recording_ids)

    def test_item_is_two_band_normalized_window_with_manifest_label(self):
        dataset = DroneRFWindowDataset(DATASET_ROOT, split="validation")
        signal, label, index = dataset[0]
        self.assertEqual(tuple(signal.shape), (2, 4096))
        self.assertEqual(str(signal.dtype), "torch.float32")
        self.assertEqual(index, 0)
        metadata = dataset.sample_metadata(index)
        self.assertEqual(label, DRONE_TYPE_LABELS[metadata.drone_type])
        values = signal.numpy()
        np.testing.assert_allclose(values.mean(axis=1), [0, 0], atol=1e-5)
        np.testing.assert_allclose(values.std(axis=1), [1, 1], atol=1e-5)

    def test_data_identity_is_backed_by_verified_hashes(self):
        dataset = DroneRFWindowDataset(DATASET_ROOT, split="test", normalization="none")
        self.assertEqual(dataset.data_identity["dataset"], "DroneRF")
        self.assertEqual(dataset.data_identity["split_seed"], 20260908)
        self.assertEqual(len(dataset.data_identity["verification_sha256"]), 64)

    def test_low_24_selection_preserves_source_channel_and_changes_identity(self):
        two_band = DroneRFWindowDataset(
            DATASET_ROOT, split="validation", normalization="none"
        )
        low_band = DroneRFWindowDataset(
            DATASET_ROOT,
            split="validation",
            normalization="none",
            band_selection="low_2_4ghz",
        )
        two_band_signal, two_band_label, _ = two_band[0]
        low_band_signal, low_band_label, _ = low_band[0]
        self.assertEqual(tuple(low_band_signal.shape), (1, 4096))
        self.assertEqual(low_band_label, two_band_label)
        np.testing.assert_array_equal(low_band_signal.numpy()[0], two_band_signal.numpy()[0])
        self.assertEqual(low_band.channels, 1)
        self.assertEqual(low_band.source_channel_indices, (0,))
        self.assertEqual(
            low_band.data_identity["input_semantics"],
            "lower_half_of_2_4ghz_real_time_domain_amplitude",
        )
        self.assertNotEqual(low_band.data_identity, two_band.data_identity)

    def test_rejects_unknown_band_selection(self):
        with self.assertRaisesRegex(ValueError, "频段选择"):
            DroneRFWindowDataset(
                DATASET_ROOT, split="train", band_selection="unverified_band"
            )


if __name__ == "__main__":
    unittest.main()
