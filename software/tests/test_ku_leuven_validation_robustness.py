from pathlib import Path
import unittest

import numpy as np

from training.common import TrainingConfigError
from training.evaluate_ku_leuven_robustness import (
    ENSEMBLE_ID,
    ensemble_probabilities,
    evaluate_probability_sets,
    expand_conditions,
    load_protocol,
    paired_bootstrap_clean_delta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "training"
    / "configs"
    / "ku_leuven_tcn_local_validation_robustness_v1.json"
)


class KULeuvenValidationRobustnessTests(unittest.TestCase):
    def test_repository_protocol_is_validation_only_and_expands_forty_conditions(self):
        protocol = load_protocol(PROTOCOL_PATH)
        conditions = expand_conditions(protocol)
        self.assertEqual(len(conditions), 40)
        self.assertEqual(len({item["condition_id"] for item in conditions}), 40)
        self.assertEqual(conditions[0]["condition_id"], "clean")
        self.assertFalse(protocol["data_access"]["test_data_allowed"])
        self.assertFalse(protocol["data_access"]["unknown_data_allowed"])
        self.assertFalse(protocol["data_access"]["a800_protocol_replaced"])

    def test_ensemble_is_arithmetic_mean_of_three_probability_sets(self):
        members = np.asarray(
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.6, 0.4], [0.4, 0.6]],
                [[0.7, 0.3], [0.2, 0.8]],
            ],
            dtype=np.float64,
        )
        result = ensemble_probabilities(members)
        np.testing.assert_allclose(result, [[0.7, 0.3], [0.3, 0.7]])

    def test_ensemble_rejects_wrong_member_count(self):
        with self.assertRaises(TrainingConfigError):
            ensemble_probabilities(np.full((2, 4, 3), 1 / 3))

    def test_probability_evaluation_enforces_recording_window_count(self):
        probabilities = {
            f"model_{index}": np.asarray(
                [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
                dtype=np.float64,
            )
            for index in range(3)
        }
        metrics, cache = evaluate_probability_sets(
            probabilities,
            np.asarray([0, 0, 1, 1]),
            ["r1", "r1", "r2", "r2"],
            {"a": 0, "b": 1},
            expected_windows_per_recording=2,
        )
        self.assertEqual(metrics[ENSEMBLE_ID]["recording"]["macro_f1"], 1.0)
        self.assertEqual(cache[ENSEMBLE_ID]["recording_ids"], ("r1", "r2"))
        with self.assertRaises(ValueError):
            evaluate_probability_sets(
                probabilities,
                np.asarray([0, 0, 1, 1]),
                ["r1", "r1", "r2", "r2"],
                {"a": 0, "b": 1},
                expected_windows_per_recording=3,
            )

    def test_paired_bootstrap_reports_condition_minus_clean(self):
        targets = np.asarray([0, 0, 1, 1])
        clean = np.asarray([0, 0, 1, 1])
        condition = np.asarray([0, 1, 1, 1])
        indices = np.asarray([[0, 1, 2, 3], [1, 1, 2, 2]] * 50)
        report = paired_bootstrap_clean_delta(
            targets,
            condition,
            clean,
            indices,
            num_classes=2,
        )
        self.assertLess(report["point"], 0.0)
        self.assertEqual(len(report["ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
