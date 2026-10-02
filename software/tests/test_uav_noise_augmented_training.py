import json
import unittest
from pathlib import Path

from airwatch.data.data_provenance import sha256_file
from training.common import load_config
from training.uav_baseline import train_uav


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "training/configs/dronerf_tcn_noise_augmented_validation_v1.json"
PROTOCOL_PATH = ROOT / "training/configs/dronerf_tcn_noise_augmented_protocol_v1.json"
COMPARISON_PROTOCOL_PATH = ROOT / "training/configs/dronerf_tcn_noise_augmented_validation_comparison_protocol_v1.json"


class UAVNoiseAugmentedTrainingTests(unittest.TestCase):
    def test_protocol_freezes_candidate_and_validation_only_selection(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sha256_file(CONFIG_PATH), protocol["candidate_config_sha256"])
        self.assertEqual(config["validation_robustness"]["snr_db_levels"], [0, 5, 10, 15, 20])
        self.assertEqual(config["training_augmentation"]["noise_probability"], 0.75)
        self.assertEqual(protocol["selection"]["split"], "validation")
        self.assertIn("Do not evaluate", protocol["test_policy"])
        reference = protocol["reference"]
        self.assertEqual(
            sha256_file(ROOT / reference["training_evidence"]),
            reference["training_evidence_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / reference["checkpoint"]), reference["checkpoint_sha256"]
        )

    def test_dry_run_records_augmentation_and_guardrail_without_writing(self):
        result = train_uav(load_config(CONFIG_PATH), dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["training_augmentation"]["noise_seed"], 2026090801)
        self.assertEqual(
            result["validation_robustness"]["clean_guardrail_min_recording_macro_f1"],
            0.9065079365079365,
        )
        self.assertIn("no test evaluation", result["selection_rule"])

    def test_comparison_protocol_is_validation_only_and_hash_frozen(self):
        protocol = json.loads(COMPARISON_PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["split"], "validation")
        self.assertIn("not independent", protocol["test_usage"])
        self.assertEqual([item["id"] for item in protocol["models"]], [
            "tcn_baseline", "tcn_noise_augmented"
        ])
        for model in protocol["models"]:
            self.assertEqual(sha256_file(ROOT / model["config"]), model["config_sha256"])
            self.assertEqual(
                sha256_file(ROOT / model["checkpoint"]), model["checkpoint_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
