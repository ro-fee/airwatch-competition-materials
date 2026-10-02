"""Behavior tests for deterministic bearing-signal perturbations."""

from __future__ import annotations

import unittest

import torch

from airwatch.analysis.bearing_robustness import (
    BearingPerturbation,
    apply_bearing_perturbation,
)


class BearingPerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32).unsqueeze(0)

    def test_clean_condition_preserves_values_without_aliasing_input(self) -> None:
        original = self.window.clone()
        result = apply_bearing_perturbation(
            self.window,
            BearingPerturbation("clean", "clean", None, "Clean test signal"),
            seed=20260903,
            sample_index=0,
        )
        self.assertTrue(torch.equal(result, original))
        self.assertNotEqual(result.data_ptr(), self.window.data_ptr())
        result[0, 0] = 99.0
        self.assertTrue(torch.equal(self.window, original))

    def test_awgn_is_reproducible_and_reaches_requested_snr(self) -> None:
        condition = BearingPerturbation("awgn_10db", "awgn_snr_db", 10.0, "10 dB AWGN")
        first = apply_bearing_perturbation(
            self.window, condition, seed=1234, sample_index=7
        )
        second = apply_bearing_perturbation(
            self.window, condition, seed=1234, sample_index=7
        )
        other_sample = apply_bearing_perturbation(
            self.window, condition, seed=1234, sample_index=8
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_sample))

        noise = first - self.window
        signal_power = torch.mean(self.window.square()).item()
        noise_power = torch.mean(noise.square()).item()
        measured_snr = 10.0 * torch.log10(
            torch.tensor(signal_power / noise_power)
        ).item()
        self.assertAlmostEqual(measured_snr, 10.0, delta=0.15)

    def test_gain_and_circular_shift_have_explicit_effects(self) -> None:
        gained = apply_bearing_perturbation(
            self.window,
            BearingPerturbation("gain_half", "amplitude_scale", 0.5, "Half gain"),
            seed=0,
            sample_index=0,
        )
        shifted = apply_bearing_perturbation(
            self.window,
            BearingPerturbation("shift_128", "circular_shift_samples", 128, "Shift"),
            seed=0,
            sample_index=0,
        )
        self.assertTrue(torch.equal(gained, self.window * 0.5))
        self.assertTrue(torch.equal(shifted, torch.roll(self.window, shifts=128, dims=-1)))

    def test_time_scale_preserves_shape_and_finite_values(self) -> None:
        scaled = apply_bearing_perturbation(
            self.window,
            BearingPerturbation("speed_95pct", "time_scale", 0.95, "Speed drift"),
            seed=0,
            sample_index=0,
        )
        self.assertEqual(scaled.shape, self.window.shape)
        self.assertTrue(torch.isfinite(scaled).all())
        self.assertFalse(torch.equal(scaled, self.window))

    def test_invalid_condition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported perturbation kind"):
            apply_bearing_perturbation(
                self.window,
                BearingPerturbation("bad", "mystery", 1.0, "Bad"),  # type: ignore[arg-type]
                seed=0,
                sample_index=0,
            )


if __name__ == "__main__":
    unittest.main()
