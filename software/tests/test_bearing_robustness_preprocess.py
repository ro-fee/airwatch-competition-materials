"""Tests for the read-only bearing preprocessing-order robustness audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from airwatch.analysis import BearingPerturbation
from training.audit_bearing_robustness_preprocess import (
    NORMALIZE_THEN_PERTURB,
    PERTURB_THEN_NORMALIZE,
    _ensure_outputs_absent,
    _output_paths,
    apply_preprocess_order,
)
from training.common import TrainingConfigError


class BearingPreprocessOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260903)
        self.raw = torch.randn((1, 1024), generator=generator, dtype=torch.float32)
        self.clean = BearingPerturbation("clean", "clean", None, "Clean")

    def test_clean_condition_produces_the_same_model_input_in_both_orders(self) -> None:
        original = self.raw.clone()
        first = apply_preprocess_order(
            self.raw,
            self.clean,
            order=NORMALIZE_THEN_PERTURB,
            normalization="window_zscore",
            seed=1,
            sample_index=2,
        )
        second = apply_preprocess_order(
            self.raw,
            self.clean,
            order=PERTURB_THEN_NORMALIZE,
            normalization="window_zscore",
            seed=1,
            sample_index=2,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(self.raw, original))
        self.assertNotEqual(first.data_ptr(), self.raw.data_ptr())

    def test_gain_before_window_zscore_is_effectively_cancelled(self) -> None:
        gain = BearingPerturbation("gain_half", "amplitude_scale", 0.5, "Half gain")
        clean_input = apply_preprocess_order(
            self.raw,
            self.clean,
            order=PERTURB_THEN_NORMALIZE,
            normalization="window_zscore",
            seed=0,
            sample_index=0,
        )
        realistic_input = apply_preprocess_order(
            self.raw,
            gain,
            order=PERTURB_THEN_NORMALIZE,
            normalization="window_zscore",
            seed=0,
            sample_index=0,
        )
        self.assertTrue(torch.allclose(clean_input, realistic_input, atol=1e-6, rtol=1e-6))

    def test_gain_after_window_zscore_changes_model_input_scale(self) -> None:
        gain = BearingPerturbation("gain_double", "amplitude_scale", 2.0, "Double gain")
        clean_input = apply_preprocess_order(
            self.raw,
            self.clean,
            order=NORMALIZE_THEN_PERTURB,
            normalization="window_zscore",
            seed=0,
            sample_index=0,
        )
        legacy_input = apply_preprocess_order(
            self.raw,
            gain,
            order=NORMALIZE_THEN_PERTURB,
            normalization="window_zscore",
            seed=0,
            sample_index=0,
        )
        self.assertTrue(torch.allclose(legacy_input, clean_input * 2.0))
        self.assertGreater(float(torch.mean(torch.abs(legacy_input - clean_input))), 0.5)

    def test_awgn_is_reproducible_for_each_order(self) -> None:
        condition = BearingPerturbation("awgn_5db", "awgn_snr_db", 5.0, "5 dB")
        for order in (NORMALIZE_THEN_PERTURB, PERTURB_THEN_NORMALIZE):
            first = apply_preprocess_order(
                self.raw,
                condition,
                order=order,
                normalization="window_zscore",
                seed=123,
                sample_index=7,
            )
            second = apply_preprocess_order(
                self.raw,
                condition,
                order=order,
                normalization="window_zscore",
                seed=123,
                sample_index=7,
            )
            self.assertTrue(torch.equal(first, second))

    def test_audit_rejects_non_window_zscore_normalization(self) -> None:
        with self.assertRaisesRegex(TrainingConfigError, "window_zscore"):
            apply_preprocess_order(
                self.raw,
                self.clean,
                order=PERTURB_THEN_NORMALIZE,
                normalization="none",
                seed=0,
                sample_index=0,
            )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _output_paths(Path(temporary_directory), "safe_audit_run")
            paths["evidence"].touch()
            with self.assertRaisesRegex(TrainingConfigError, "refusing to overwrite"):
                _ensure_outputs_absent(paths)


if __name__ == "__main__":
    unittest.main()
