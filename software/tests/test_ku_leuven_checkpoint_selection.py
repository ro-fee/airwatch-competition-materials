"""Tests for the clean-guarded KU Leuven V3 checkpoint rule."""

from __future__ import annotations

import unittest

from training.common import TrainingConfigError
from training.ku_leuven_checkpoint_selection import (
    EXPECTED_REPEAT_SEEDS,
    PROTOCOL_ID,
    is_checkpoint_selection_improvement,
    next_stale_epochs,
    summarize_checkpoint_selection,
    validate_checkpoint_selection,
)
from training.run_ku_leuven_worst_noise_selection_local import (
    compare,
    load_v3_protocol,
)
from training.run_ku_leuven_weighted_worst_noise_local import load_v4_protocol


DEFINITION = {
    "protocol": PROTOCOL_ID,
    "clean_recording_macro_f1_floor": 0.97,
    "awgn_snr_db": -5.0,
    "repeat_seeds": list(EXPECTED_REPEAT_SEEDS),
}


def validation(score: float, loss: float = 0.4) -> dict:
    return {"loss": loss, "recording": {"macro_f1": score}}


class KULeuvenCheckpointSelectionTests(unittest.TestCase):
    def test_repository_v3_protocol_preserves_data_boundaries(self):
        protocol = load_v3_protocol()
        self.assertTrue(protocol["development_validation_only"])
        self.assertFalse(protocol["test_data_allowed"])
        self.assertFalse(protocol["unknown_data_allowed"])
        self.assertFalse(protocol["open_set_threshold_calibration_allowed"])
        self.assertFalse(protocol["a800_protocol_replaced"])

    def test_repository_v4_changes_only_weighted_training_protocol(self):
        protocol = load_v4_protocol()
        self.assertEqual(
            protocol["augmentation"]["protocol"],
            "ku-leuven-clean-weighted-minus5-awgn-multipath-training-v3",
        )
        self.assertEqual(protocol["checkpoint_selection"], DEFINITION)
        self.assertFalse(protocol["test_data_allowed"])
        self.assertFalse(protocol["unknown_data_allowed"])
        self.assertFalse(protocol["a800_protocol_replaced"])

    def test_frozen_definition_rejects_changed_stress_condition(self):
        self.assertEqual(validate_checkpoint_selection(DEFINITION), DEFINITION)
        changed = dict(DEFINITION, awgn_snr_db=0.0)
        with self.assertRaisesRegex(TrainingConfigError, "-5 dB"):
            validate_checkpoint_selection(changed)

    def test_clean_guard_blocks_stress_score_from_becoming_checkpoint(self):
        result = summarize_checkpoint_selection(
            validation(0.96),
            [validation(0.9), validation(0.9), validation(0.9)],
            DEFINITION,
        )
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["selection_rank"])
        self.assertFalse(is_checkpoint_selection_improvement(result, None))

    def test_patience_starts_only_after_first_eligible_checkpoint(self):
        self.assertEqual(
            next_stale_epochs(
                0, improved=False, has_eligible_checkpoint=False
            ),
            0,
        )
        self.assertEqual(
            next_stale_epochs(
                3, improved=False, has_eligible_checkpoint=True
            ),
            4,
        )
        self.assertEqual(
            next_stale_epochs(
                4, improved=True, has_eligible_checkpoint=True
            ),
            0,
        )

    def test_worst_repeat_precedes_higher_mean_and_loss_tiebreak(self):
        stable = summarize_checkpoint_selection(
            validation(1.0, 0.5),
            [validation(0.5), validation(0.5), validation(0.5)],
            DEFINITION,
        )
        unstable = summarize_checkpoint_selection(
            validation(1.0, 0.2),
            [validation(0.49), validation(0.9), validation(0.9)],
            DEFINITION,
        )
        lower_loss = summarize_checkpoint_selection(
            validation(1.0, 0.3),
            [validation(0.5), validation(0.5), validation(0.5)],
            DEFINITION,
        )
        self.assertTrue(is_checkpoint_selection_improvement(stable, unstable))
        self.assertTrue(is_checkpoint_selection_improvement(lower_loss, stable))

    def test_v3_acceptance_requires_improved_minus5_mean_and_worst_case(self):
        members = ["tcn_seed20260909", "tcn_seed20260910", "tcn_seed20260911"]
        ensemble = "tcn_three_seed_probability_ensemble"

        def condition(identifier, family, value, member_scores):
            scores = dict(zip(members, member_scores))
            scores[ensemble] = sum(member_scores) / len(member_scores)
            return {
                "condition_id": identifier,
                "family": family,
                "value": value,
                "metrics": {
                    name: {"recording": {"macro_f1": score}}
                    for name, score in scores.items()
                },
            }

        prior_conditions = [
            condition("clean", "clean", "clean", [1.0, 1.0, 1.0]),
            condition("minus5", "awgn", -5.0, [0.2, 0.4, 0.6]),
            condition("zero", "awgn", 0.0, [0.8, 0.8, 0.8]),
            condition("five", "awgn", 5.0, [0.9, 0.9, 0.9]),
            condition("multipath", "multipath", "mild", [0.7, 0.7, 0.7]),
        ]
        candidate_conditions = [
            condition("clean", "clean", "clean", [0.99, 0.99, 0.99]),
            condition("minus5", "awgn", -5.0, [0.25, 0.5, 0.65]),
            condition("zero", "awgn", 0.0, [0.79, 0.79, 0.79]),
            condition("five", "awgn", 5.0, [0.89, 0.89, 0.89]),
            condition("multipath", "multipath", "mild", [0.69, 0.69, 0.69]),
        ]
        common = {
            "dataset": {"identity": "same"},
            "sources": [{"id": name} for name in members],
            "ensemble": {"id": ensemble},
        }
        result = compare(
            {**common, "conditions": prior_conditions},
            {**common, "conditions": candidate_conditions},
        )
        self.assertTrue(result["acceptance_passed"])
        self.assertAlmostEqual(
            result["minus5_worst_seed_repeat"]["difference"], 0.05
        )

        candidate_conditions[1] = condition(
            "minus5", "awgn", -5.0, [0.2, 0.7, 0.7]
        )
        failed = compare(
            {**common, "conditions": prior_conditions},
            {**common, "conditions": candidate_conditions},
        )
        self.assertFalse(failed["acceptance_passed"])
        self.assertFalse(failed["acceptance_checks"]["minus5_worst_case_improved"])


if __name__ == "__main__":
    unittest.main()
