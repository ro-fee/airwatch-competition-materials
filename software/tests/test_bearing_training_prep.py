import json
import tempfile
import unittest
from pathlib import Path

from training.common import (
    TrainingConfigError,
    ensure_output_paths_available,
    load_config,
    project_root_from_config,
    resolve_output_paths,
    resolve_run_name,
)
from training.train import train


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training" / "configs" / "bearing_cnn_baseline.json"


class TestBearingTrainingPreparation(unittest.TestCase):
    def test_dry_run_validates_data_and_model_wiring_without_outputs(self):
        config = load_config(CONFIG)
        output_dirs = [
            ROOT / "artifacts" / "checkpoints" / "bearing",
            ROOT / "artifacts" / "evidence" / "bearing",
        ]

        def snapshot(paths):
            snapshot = {}
            for path in paths:
                snapshot[str(path)] = {
                    "exists": path.exists(),
                    "files": {
                        str(item.relative_to(path)): (
                            item.stat().st_size,
                            item.stat().st_mtime_ns,
                        )
                        for item in path.rglob("*")
                        if item.is_file()
                    }
                    if path.is_dir()
                    else {},
                }
            return snapshot

        before_outputs = snapshot(output_dirs)
        result = train(config, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["dataset"]["train_windows"], 1422)
        self.assertEqual(result["dataset"]["validation_windows"], 828)
        self.assertEqual(result["dataset"]["test_windows"], 831)
        self.assertEqual(result["model"]["input_shape"], [1, 1024])
        self.assertEqual(result["model"]["num_classes"], 4)
        self.assertIn(result["environment"]["device"], {"cuda", "cpu"})
        self.assertEqual(snapshot(output_dirs), before_outputs)

    def test_config_paths_are_project_relative(self):
        config = load_config(CONFIG)
        self.assertEqual(project_root_from_config(config), ROOT)
        self.assertEqual(config["dataset"]["normalization"], "none")
        self.assertEqual(config["training"]["num_workers"], 0)

    def test_run_name_controls_new_output_paths_without_leaving_project(self):
        config = load_config(CONFIG)
        paths = resolve_output_paths(config, ROOT, "comparison_cnn_20260903")
        self.assertTrue(all(path.parent.is_relative_to(ROOT) for path in paths.values()))
        self.assertEqual(
            paths["checkpoint"].name,
            "comparison_cnn_20260903_best.pt",
        )
        self.assertEqual(
            paths["test_predictions"].name,
            "comparison_cnn_20260903_test_predictions.csv",
        )

    def test_invalid_run_names_are_rejected(self):
        config = load_config(CONFIG)
        invalid_names = ["", "   ", "../escape", r"..\escape", "run name", "run:name"]
        for invalid_name in invalid_names:
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(TrainingConfigError):
                    resolve_run_name(config, invalid_name)

    def test_existing_outputs_are_never_overwritten(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = dict(config)
            config["output"] = {
                "checkpoint_dir": "checkpoints",
                "evidence_dir": "evidence",
            }
            paths = resolve_output_paths(config, root, "protected_run")
            paths["checkpoint"].parent.mkdir(parents=True)
            paths["checkpoint"].write_bytes(b"old checkpoint")
            with self.assertRaisesRegex(TrainingConfigError, "refusing to overwrite"):
                ensure_output_paths_available(paths)
            self.assertEqual(paths["checkpoint"].read_bytes(), b"old checkpoint")

    def test_default_formal_training_refuses_existing_baseline_outputs(self):
        config = load_config(CONFIG)
        with self.assertRaisesRegex(TrainingConfigError, "Choose a new --run-name"):
            train(config, dry_run=False)


if __name__ == "__main__":
    unittest.main()
