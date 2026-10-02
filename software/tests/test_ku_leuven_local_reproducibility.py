import copy
import json
from pathlib import Path
import tempfile
import unittest

from training.common import TrainingConfigError
from training.run_ku_leuven_local_matrix import (
    load_matrix,
    summarize_candidate_runs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    PROJECT_ROOT
    / "training"
    / "configs"
    / "ku_leuven_local_reproducibility_v1.json"
)


class KULeuvenLocalReproducibilityTests(unittest.TestCase):
    @staticmethod
    def _runs():
        rows = []
        for model, scores, parameters in (
            ("DroneRFTCN", [0.80, 0.82, 0.81], 800000),
            ("DroneRFResNet18", [0.84, 0.83, 0.82], 900000),
        ):
            for seed, score in zip((20260909, 20260910, 20260911), scores):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "validation_recording_macro_f1": score,
                        "parameter_count": parameters,
                    }
                )
        return rows

    def test_repository_matrix_preserves_sealed_data_and_a800(self):
        matrix = load_matrix(MATRIX_PATH)
        self.assertFalse(matrix["comparison"]["test_data_allowed"])
        self.assertFalse(matrix["comparison"]["unknown_data_allowed"])
        self.assertFalse(matrix["comparison"]["formal_model_selection_claim_allowed"])
        self.assertFalse(matrix["a800_protocol_replaced"])

    def test_rejects_matrix_that_allows_test_data(self):
        matrix = load_matrix(MATRIX_PATH)
        changed = copy.deepcopy(matrix)
        changed["comparison"]["test_data_allowed"] = True
        changed.pop("_matrix_path")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe-matrix.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(TrainingConfigError):
                load_matrix(path)

    def test_summarizes_three_seed_development_leader(self):
        summaries, leader = summarize_candidate_runs(self._runs())
        self.assertEqual(len(summaries), 2)
        self.assertEqual(leader["model"], "DroneRFResNet18")
        self.assertAlmostEqual(leader["mean_validation_recording_macro_f1"], 0.83)

    def test_tie_breaks_by_variance_then_parameter_count(self):
        runs = self._runs()
        for row in runs:
            row["validation_recording_macro_f1"] = 0.8
        _, leader = summarize_candidate_runs(runs)
        self.assertEqual(leader["model"], "DroneRFTCN")

    def test_rejects_incomplete_seed_matrix(self):
        with self.assertRaises(TrainingConfigError):
            summarize_candidate_runs(self._runs()[:-1])


if __name__ == "__main__":
    unittest.main()
