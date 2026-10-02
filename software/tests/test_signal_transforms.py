"""Unit tests for the UI-independent signal plot transformations."""

from __future__ import annotations

import unittest

import numpy as np

from airwatch.analysis.signal_transforms import (
    SignalViewError,
    compute_constellation,
    compute_spectrogram,
    compute_spectrum,
    compute_waveform,
    prepare_signal,
)


class SignalTransformTests(unittest.TestCase):
    def test_real_signal_is_canonicalized_without_mutation(self) -> None:
        source = np.arange(8, dtype=np.float32)
        prepared = prepare_signal(source)

        self.assertEqual(prepared.kind, "real")
        self.assertEqual(prepared.source_shape, (8,))
        self.assertEqual(prepared.sample_count, 8)
        self.assertFalse(prepared.samples.flags.writeable)
        source[0] = 100
        self.assertEqual(prepared.samples[0], 0)

    def test_two_channel_matrix_becomes_complex_iq(self) -> None:
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        prepared = prepare_signal(matrix)

        np.testing.assert_array_equal(prepared.samples, [1 + 4j, 2 + 5j, 3 + 6j])
        self.assertEqual(prepared.kind, "iq")

        transposed = prepare_signal(matrix.T)
        np.testing.assert_array_equal(transposed.samples, prepared.samples)

    def test_waveform_returns_time_axis_and_iq_channels(self) -> None:
        view = compute_waveform(
            np.array([1 + 2j, 3 + 4j, 5 + 6j]),
            sample_rate_hz=10,
        )

        np.testing.assert_allclose(view.x, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(view.values[0], [1, 3, 5])
        np.testing.assert_allclose(view.values[1], [2, 4, 6])
        self.assertEqual(view.channel_labels, ("I", "Q"))
        self.assertEqual(view.x_label, "时间 (s)")

    def test_real_spectrum_is_one_sided_and_iq_spectrum_is_centered(self) -> None:
        real = compute_spectrum(np.cos(2 * np.pi * np.arange(8) / 8), sample_rate_hz=8)
        self.assertFalse(real.two_sided)
        self.assertEqual(real.frequency_hz[0], 0.0)
        self.assertEqual(real.frequency_hz[-1], 4.0)

        iq = compute_spectrum(np.exp(2j * np.pi * np.arange(8) / 8), sample_rate_hz=8)
        self.assertTrue(iq.two_sided)
        self.assertAlmostEqual(iq.frequency_hz[0], -4.0)
        self.assertAlmostEqual(iq.frequency_hz[-1], 3.0)

    def test_spectrogram_has_matching_axes_and_does_not_silently_shrink_window(self) -> None:
        samples = np.sin(2 * np.pi * 4 * np.arange(64) / 64)
        view = compute_spectrogram(samples, sample_rate_hz=64, nperseg=16)

        self.assertEqual(view.power.shape, (9, 7))
        self.assertEqual(view.power_db.shape, view.power.shape)
        self.assertEqual(view.frequency_hz.shape, (9,))
        self.assertEqual(view.time_s.shape, (7,))
        self.assertFalse(view.two_sided)
        with self.assertRaisesRegex(SignalViewError, "cannot exceed"):
            compute_spectrogram(samples, sample_rate_hz=64, nperseg=65)

    def test_constellation_requires_explicit_iq_or_analytic_compatibility(self) -> None:
        with self.assertRaisesRegex(SignalViewError, "requires IQ"):
            compute_constellation(np.arange(8, dtype=float))

        iq_view = compute_constellation(np.array([1 + 1j, -1 - 1j]), max_points=1)
        self.assertEqual(iq_view.point_count, 1)
        self.assertEqual(iq_view.source_kind, "iq")

        analytic = compute_constellation(
            np.cos(2 * np.pi * np.arange(32) / 8),
            analytic_from_real=True,
        )
        self.assertEqual(analytic.source_kind, "analytic_real")
        self.assertEqual(analytic.point_count, 32)

    def test_invalid_inputs_are_rejected(self) -> None:
        for value in (0, -1, float("nan")):
            with self.subTest(sample_rate=value):
                with self.assertRaises(SignalViewError):
                    compute_spectrum([1, 2, 3], sample_rate_hz=value)
        with self.assertRaisesRegex(SignalViewError, "exactly two channels"):
            prepare_signal(np.zeros((3, 4)))
        with self.assertRaisesRegex(SignalViewError, "NaN"):
            prepare_signal(np.array([1.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
