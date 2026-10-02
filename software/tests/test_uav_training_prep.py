import unittest
from pathlib import Path

from training.common import load_config
from training.uav_baseline import build_uav_datasets, train_uav


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training" / "configs" / "dronerf_cnn_type_baseline_v1.json"
LOW24_CONFIG = ROOT / "training" / "configs" / "dronerf_tcn_low24_type_baseline_v1.json"


class UAVTrainingPreparationTests(unittest.TestCase):
    def test_dry_run_validates_fixed_data_model_and_outputs_without_writing(self):
        config = load_config(CONFIG)
        result = train_uav(config, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["dataset"]["train_windows"], 5152)
        self.assertEqual(result["dataset"]["validation_windows"], 1056)
        self.assertEqual(result["dataset"]["test_windows"], 1056)
        self.assertEqual(result["dataset"]["train_recordings"], 161)
        self.assertEqual(result["model"]["input_shape"], [2, 4096])
        self.assertEqual(result["model"]["num_classes"], 4)
        self.assertIn(result["environment"]["device"], {"cuda", "cpu"})

    def test_three_splits_share_identity_and_no_recording(self):
        config = load_config(CONFIG)
        datasets = build_uav_datasets(config, ROOT)
        self.assertEqual(datasets["train"].data_identity, datasets["test"].data_identity)
        self.assertFalse(datasets["train"].recording_ids & datasets["test"].recording_ids)

    def test_low_24_dry_run_binds_single_channel_semantics(self):
        config = load_config(LOW24_CONFIG)
        result = train_uav(config, dry_run=True)
        self.assertEqual(result["model"]["input_shape"], [1, 4096])
        self.assertEqual(result["dataset"]["channels"], 1)
        self.assertEqual(
            result["dataset"]["data_identity"]["input_semantics"],
            "lower_half_of_2_4ghz_real_time_domain_amplitude",
        )
        self.assertEqual(
            result["dataset"]["data_identity"]["source_channel_indices"], [0]
        )


if __name__ == "__main__":
    unittest.main()
