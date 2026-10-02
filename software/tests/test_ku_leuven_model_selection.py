import unittest

from training.common import TrainingConfigError
from training.select_ku_leuven_a800_model import select_candidate_summaries


class KULeuvenModelSelectionTests(unittest.TestCase):
    @staticmethod
    def _runs():
        rows = []
        for model, scores, parameters in (
            ("DroneRFTCN", [0.80, 0.82, 0.81], 800000),
            ("DroneRFResNet18", [0.84, 0.83, 0.82], 3000000),
        ):
            for seed, score in zip((20260909, 20260910, 20260911), scores):
                rows.append({
                    "model": model, "seed": seed,
                    "validation_recording_macro_f1": score,
                    "parameter_count": parameters,
                })
        return rows

    def test_selects_highest_three_seed_mean(self):
        summaries, winner = select_candidate_summaries(self._runs())
        self.assertEqual(len(summaries), 2)
        self.assertEqual(winner["model"], "DroneRFResNet18")
        self.assertAlmostEqual(winner["mean_validation_recording_macro_f1"], 0.83)

    def test_tie_breaks_by_variance_then_parameter_count(self):
        runs = self._runs()
        for row in runs:
            row["validation_recording_macro_f1"] = 0.8
        _, winner = select_candidate_summaries(runs)
        self.assertEqual(winner["model"], "DroneRFTCN")

    def test_rejects_incomplete_seed_matrix(self):
        with self.assertRaises(TrainingConfigError):
            select_candidate_summaries(self._runs()[:-1])


if __name__ == "__main__":
    unittest.main()
