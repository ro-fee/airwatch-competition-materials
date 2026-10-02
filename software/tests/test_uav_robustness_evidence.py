import csv
import json
import tempfile
import unittest
from pathlib import Path

from airwatch.data.data_provenance import sha256_file
from airwatch.evidence.uav_robustness import build_uav_robustness_failure_analysis


class UAVRobustnessEvidenceTests(unittest.TestCase):
    def test_builds_recording_and_window_failure_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.csv"
            rows = []
            for model in ("base", "candidate"):
                for recording, target in (("r0", 0), ("r1", 1)):
                    for window in range(2):
                        prediction = target if model == "candidate" else 0
                        rows.append({
                            "condition": "0_db", "model_id": model,
                            "recording_id": recording, "target": target,
                            "prediction": prediction, "prob_a": 0.9 if prediction == 0 else 0.1,
                            "prob_b": 0.1 if prediction == 0 else 0.9,
                        })
            with prediction_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            summary = {
                "status": "completed", "label_map": {"a": 0, "b": 1},
                "results": [{"condition": "0_db"}],
                "sources": [{"id": "base"}, {"id": "candidate"}],
                "window_count_per_condition": 4, "recording_count_per_condition": 2,
                "data_identity": {"hash": "same"}, "interpretation_boundary": "synthetic test",
                "artifacts": {"predictions_csv": str(prediction_path),
                              "predictions_csv_sha256": sha256_file(prediction_path)},
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report = build_uav_robustness_failure_analysis(
                summary_path, output_json=root / "failures.json", output_csv=root / "failures.csv"
            )
            candidate = next(item for item in report["analyses"] if item["model_id"] == "candidate")
            self.assertEqual(candidate["recording"]["accuracy"], 1.0)
            base = next(item for item in report["analyses"] if item["model_id"] == "base")
            self.assertEqual(base["window_prediction_collapse"]["dominant_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
