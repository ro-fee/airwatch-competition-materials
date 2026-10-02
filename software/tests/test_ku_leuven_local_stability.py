import unittest

from training.common import TrainingConfigError
from training.analyze_ku_leuven_local_stability import summarize_run_history


def _payload():
    def epoch(number, recording_f1, window_f1, loss):
        return {
            "epoch": number,
            "validation": {
                "loss": loss,
                "window": {"macro_f1": window_f1, "accuracy": window_f1},
                "recording": {
                    "macro_f1": recording_f1,
                    "per_class": {
                        "a": {"recall": 1.0},
                        "b": {"recall": 1.0},
                    },
                },
            },
        }

    return {
        "model": {"name": "DroneRFTCN"},
        "seed": 20260909,
        "run_name": "example",
        "best_validation_recording_macro_f1": 1.0,
        "best_validation_window_loss": 0.4,
        "history": [
            epoch(1, 0.5, 0.6, 0.8),
            epoch(2, 1.0, 0.7, 0.5),
            epoch(3, 1.0, 0.8, 0.4),
        ],
    }


class KULeuvenLocalStabilityTests(unittest.TestCase):
    def test_reconstructs_tie_broken_checkpoint(self):
        summary = summarize_run_history(_payload())
        self.assertEqual(summary["selected_checkpoint_epoch"], 3)
        self.assertEqual(summary["selected_checkpoint_window_macro_f1"], 0.8)
        self.assertEqual(summary["perfect_recording_macro_f1_epoch_count"], 2)

    def test_rejects_inconsistent_best_loss(self):
        payload = _payload()
        payload["best_validation_window_loss"] = 0.3
        with self.assertRaises(TrainingConfigError):
            summarize_run_history(payload)

    def test_rejects_empty_history(self):
        payload = _payload()
        payload["history"] = []
        with self.assertRaises(TrainingConfigError):
            summarize_run_history(payload)


if __name__ == "__main__":
    unittest.main()
