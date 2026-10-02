import json
import tempfile
import unittest
from pathlib import Path

from airwatch.evidence.uav_baselines import (
    UAVBaselineEvidenceError,
    build_uav_ablation_summary,
    build_uav_baseline_comparison,
    build_uav_multiseed_summary,
)


class UAVBaselineEvidenceTests(unittest.TestCase):
    def test_ablation_summary_selects_by_validation_and_checks_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol = {
                "protocol": "test-ablation-v1", "frozen_on": "2026-09-08", "seed": 7,
                "selection_metric": "validation recording macro_f1",
                "test_usage": "report once after validation freeze",
                "variants": [
                    {"id": "spectral", "model": "Spectral", "presence_head": False},
                    {"id": "adaptive", "model": "Dual", "fusion_mode": "adaptive",
                     "presence_head": True, "presence_loss_weight": 0.3},
                ],
                "fixed_training": {
                    "batch_size": 4, "epochs": 2, "early_stopping_patience": 1,
                    "learning_rate": 0.001, "weight_decay": 0.0001,
                    "normalization": "zscore",
                },
            }
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            identity = {"dataset": "DroneRF", "hash": "same"}
            paths = []
            variants = [
                ("spectral", {"name": "Spectral", "presence_head": False}, 0.8, 0.95),
                ("adaptive", {"name": "Dual", "fusion_mode": "adaptive",
                              "presence_head": True, "presence_loss_weight": 0.3}, 0.9, 0.70),
            ]
            for variant_id, model, validation_score, test_score in variants:
                run = f"run_{variant_id}"
                checkpoint = root / f"{run}_best.pt"; checkpoint.write_bytes(run.encode())
                config = {
                    "experiment_protocol": "training/configs/protocol.json",
                    "dataset": {"normalization": "zscore"}, "model": model,
                    "training": {"seed": 7, "batch_size": 4, "epochs": 2,
                                 "early_stopping_patience": 1, "learning_rate": 0.001,
                                 "weight_decay": 0.0001},
                }
                config_path = root / f"{run}_config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                training = {
                    "seed": 7, "model": {"name": model["name"], "parameter_count": 10},
                    "dataset": {"data_identity": identity},
                    "resolved_config_path": str(config_path),
                    "best_validation_recording_macro_f1": validation_score,
                }
                (root / f"{run}_training.json").write_text(
                    json.dumps(training), encoding="utf-8"
                )
                metrics = {
                    "window": {"accuracy": test_score, "macro_f1": test_score},
                    "recording": {"accuracy": test_score, "macro_f1": test_score},
                }
                if variant_id == "adaptive":
                    metrics["presence_recording"] = {"macro_f1": 0.88}
                    metrics["fusion_weight_time"] = {
                        "mean": 0.6, "standard_deviation": 0.1,
                    }
                test = {
                    "experiment_name": run, "status": "completed", "split": "test",
                    "data_identity": identity, "label_map": {"background": 0, "drone": 1},
                    "checkpoint_path": str(checkpoint), "checkpoint_epoch": 1,
                    "metrics": metrics, "known_limitation": "single source",
                }
                path = root / f"{run}_test.json"
                path.write_text(json.dumps(test), encoding="utf-8"); paths.append(path)
            report = build_uav_ablation_summary(
                protocol_path, paths, output_json=root / "summary.json",
                output_csv=root / "summary.csv",
            )
            self.assertEqual(report["selected_variant_by_validation"], "adaptive")
            self.assertEqual(report["results"][1]["test_presence_recording_macro_f1"], 0.88)
            self.assertEqual(report["results"][0]["test_recording_macro_f1"], 0.95)
            self.assertTrue((root / "summary.csv").is_file())

    def test_builds_ranked_comparison_only_for_same_data_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"dataset": "DroneRF", "hash": "abc"}
            evidence_paths = []
            for name, score, parameters in (("a", 0.7, 20), ("b", 0.9, 30)):
                checkpoint = root / f"{name}_best.pt"; checkpoint.write_bytes(name.encode())
                training = {
                    "seed": 7,
                    "model": {"name": name, "parameter_count": parameters},
                    "dataset": {"data_identity": identity},
                }
                (root / f"{name}_training.json").write_text(json.dumps(training), encoding="utf-8")
                test = {
                    "status": "completed", "split": "test", "data_identity": identity,
                    "label_map": {"background": 0, "drone": 1},
                    "checkpoint_path": str(checkpoint), "checkpoint_epoch": 2,
                    "known_limitation": "test limitation",
                    "metrics": {
                        "window": {"accuracy": score, "macro_f1": score,
                                   "windows_per_second": 10},
                        "recording": {"accuracy": score, "macro_f1": score},
                    },
                }
                path = root / f"{name}_test.json"
                path.write_text(json.dumps(test), encoding="utf-8")
                evidence_paths.append(path)
            report = build_uav_baseline_comparison(
                evidence_paths, output_json=root / "comparison.json",
                output_csv=root / "comparison.csv",
            )
            self.assertEqual(report["results"][0]["model"], "b")
            self.assertEqual(report["results"][0]["recording_macro_f1_rank"], 1)
            self.assertTrue((root / "comparison.csv").is_file())

    def test_rejects_mismatched_data_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(2):
                path = root / f"x{index}_test.json"
                path.write_text(json.dumps({
                    "status": "completed", "split": "test",
                    "data_identity": {"id": index}, "label_map": {"x": 0},
                }), encoding="utf-8")
                paths.append(path)
            with self.assertRaises(UAVBaselineEvidenceError):
                build_uav_baseline_comparison(
                    paths, output_json=root / "out.json", output_csv=root / "out.csv"
                )

    def test_metric_summary_uses_all_matching_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = []
            identity = {"dataset": "DroneRF", "hash": "same"}
            for model in ("m1", "m2"):
                for seed in (1, 2, 3):
                    run = f"{model}_seed{seed}"
                    checkpoint = root / f"{run}_best.pt"; checkpoint.write_bytes(run.encode())
                    (root / f"{run}_training.json").write_text(json.dumps({
                        "seed": seed, "model": {"name": model, "parameter_count": 10},
                        "dataset": {"data_identity": identity},
                    }), encoding="utf-8")
                    score = (0.5 if model == "m1" else 0.7) + seed * 0.01
                    test = {
                        "status": "completed", "split": "test", "data_identity": identity,
                        "label_map": {"a": 0, "b": 1}, "known_limitation": "limit",
                        "checkpoint_path": str(checkpoint),
                        "metrics": {
                            "window": {"accuracy": score, "macro_f1": score},
                            "recording": {"accuracy": score, "macro_f1": score},
                        },
                    }
                    path = root / f"{run}_test.json"
                    path.write_text(json.dumps(test), encoding="utf-8"); paths.append(path)
            report = build_uav_multiseed_summary(
                paths, output_json=root / "multi.json", output_csv=root / "multi.csv"
            )
            self.assertEqual(report["models"][0]["model"], "m2")
            summary = report["models"][0]["metrics"]["recording_macro_f1"]
            self.assertEqual(summary["run_count"], 3)
            self.assertAlmostEqual(summary["mean"], 0.72)
            self.assertGreater(summary["sample_standard_deviation"], 0)


if __name__ == "__main__":
    unittest.main()
