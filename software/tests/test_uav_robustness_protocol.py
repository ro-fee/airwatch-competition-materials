import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airwatch.data.data_provenance import sha256_file
from training.evaluate_uav_robustness import (
    DEFAULT_PROTOCOL,
    _paired_bootstrap,
    _recording_predictions,
    _write_csv_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


class UAVRobustnessProtocolTests(unittest.TestCase):
    def test_csv_writer_accepts_model_specific_optional_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            _write_csv_atomic(path, [{"model": "base"}, {"model": "dual", "gate": 0.7}])
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["gate"], "")
            self.assertEqual(rows[1]["gate"], "0.7")

    def test_frozen_protocol_covers_required_range_and_hashes(self):
        protocol_path = ROOT / DEFAULT_PROTOCOL
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["perturbation"]["snr_db_levels"], list(range(-20, 21, 5)))
        self.assertEqual(protocol["statistics"]["primary_unit"], "recording")
        self.assertGreaterEqual(protocol["statistics"]["bootstrap_replicates"], 2000)
        self.assertEqual(len(protocol["models"]), 2)
        for model in protocol["models"]:
            self.assertEqual(sha256_file(ROOT / model["config"]), model["config_sha256"])
            self.assertEqual(
                sha256_file(ROOT / model["checkpoint"]), model["checkpoint_sha256"]
            )

    def test_recording_predictions_average_window_probabilities(self):
        rows = [
            {"recording_id": "r1", "target": 1, "prob_a": 0.8, "prob_b": 0.2},
            {"recording_id": "r1", "target": 1, "prob_a": 0.1, "prob_b": 0.9},
        ]
        result = _recording_predictions(rows, {"a": 0, "b": 1})
        self.assertEqual(result, [{"recording_id": "r1", "target": 1, "prediction": 1}])

    def test_paired_bootstrap_detects_clear_candidate_improvement(self):
        targets = np.repeat(np.arange(4), 20)
        reference = np.roll(targets, 1)
        candidate = targets.copy()
        result = _paired_bootstrap(
            targets, {"reference": reference, "candidate": candidate},
            seed=9, replicates=500, num_classes=4,
        )
        self.assertEqual(result["difference"]["direction_supported_95pct"], "candidate_better")
        self.assertGreater(result["difference"]["ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
