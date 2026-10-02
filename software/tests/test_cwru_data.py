import tempfile
import unittest

from pathlib import Path

import numpy as np
from scipy.io import savemat

from airwatch.data import (
    AmbiguousSignalError,
    CWRUDataError,
    load_cwru_signal,
    normalize_windows,
    slice_signal,
)


ROOT = Path(__file__).resolve().parents[1]
CWRU_ROOT = ROOT / "datasets" / "bearing" / "cwru"


class TestCWRUDataAdapter(unittest.TestCase):
    def test_loads_unique_drive_end_channel_as_read_only_float32(self):
        path = CWRU_ROOT / "raw" / "normal" / "Normal_0.mat"
        record = load_cwru_signal(path)

        self.assertEqual(record.sensor_key, "X097_DE_time")
        self.assertEqual(record.signal.dtype, np.float32)
        self.assertEqual(record.signal.ndim, 1)
        self.assertGreater(record.signal.size, 0)
        self.assertTrue(np.isfinite(record.signal).all())
        self.assertFalse(record.signal.flags.writeable)

    def test_requires_explicit_channel_for_ambiguous_file(self):
        path = CWRU_ROOT / "raw" / "normal" / "Normal_2.mat"
        with self.assertRaisesRegex(AmbiguousSignalError, "X098_DE_time.*X099_DE_time"):
            load_cwru_signal(path)

        record = load_cwru_signal(path, sensor_key="X099_DE_time")
        self.assertEqual(record.sensor_key, "X099_DE_time")
        self.assertEqual(record.source_shape, (485063, 1))

    def test_zscore_is_explicit_and_does_not_change_source_file(self):
        path = CWRU_ROOT / "raw" / "normal" / "Normal_0.mat"
        raw = load_cwru_signal(path)
        zscored = load_cwru_signal(path, normalization="zscore")

        self.assertAlmostEqual(float(zscored.signal.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(zscored.signal.std()), 1.0, places=4)
        self.assertFalse(np.array_equal(raw.signal, zscored.signal))
        self.assertEqual(raw.path, zscored.path)

    def test_window_zscore_removes_per_window_offset_and_gain(self):
        base = np.linspace(-2.0, 3.0, 1024, dtype=np.float32)
        windows = np.stack((base, base * 0.5 + 7.0, base * 2.0 - 4.0))

        normalized = normalize_windows(
            windows,
            normalization="window_zscore",
        )

        np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-6)
        np.testing.assert_allclose(normalized[0], normalized[2], atol=1e-6)
        np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=1), 1.0, atol=1e-6)

    def test_window_normalization_none_returns_read_only_float32_copy(self):
        source = np.arange(12, dtype=np.float64).reshape(3, 4)

        normalized = normalize_windows(source, normalization="none")

        self.assertEqual(normalized.dtype, np.float32)
        self.assertFalse(normalized.flags.writeable)
        self.assertFalse(np.shares_memory(source, normalized))
        np.testing.assert_array_equal(normalized, source.astype(np.float32))

    def test_window_normalization_rejects_nonfinite_values(self):
        windows = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)

        with self.assertRaisesRegex(CWRUDataError, "NaN or infinite"):
            normalize_windows(windows, normalization="window_zscore")

    def test_window_zscore_rejects_constant_window(self):
        windows = np.array([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]], dtype=np.float32)

        with self.assertRaisesRegex(CWRUDataError, "constant window"):
            normalize_windows(windows, normalization="window_zscore")

    def test_window_normalization_requires_nonempty_window_matrix(self):
        for invalid in (np.arange(4, dtype=np.float32), np.empty((0, 4), dtype=np.float32)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(CWRUDataError, "non-empty 2D"):
                    normalize_windows(invalid, normalization="window_zscore")

    def test_window_zscore_must_be_applied_after_slicing(self):
        path = CWRU_ROOT / "raw" / "normal" / "Normal_0.mat"

        with self.assertRaisesRegex(CWRUDataError, "after slicing"):
            load_cwru_signal(path, normalization="window_zscore")

    def test_slice_signal_returns_complete_non_overlapping_windows(self):
        signal = np.arange(10, dtype=np.float32)
        windows = slice_signal(signal, window_size=4)

        np.testing.assert_array_equal(
            windows,
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32),
        )
        self.assertFalse(windows.flags.writeable)

    def test_slice_signal_supports_explicit_overlap_and_drops_tail(self):
        signal = np.arange(10, dtype=np.float32)
        windows = slice_signal(signal, window_size=4, step=2)

        self.assertEqual(windows.shape, (4, 4))
        np.testing.assert_array_equal(windows[-1], np.array([6, 7, 8, 9], dtype=np.float32))

    def test_short_signal_returns_empty_window_matrix(self):
        windows = slice_signal(np.arange(3, dtype=np.float32), window_size=4)
        self.assertEqual(windows.shape, (0, 4))
        self.assertEqual(windows.dtype, np.float32)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(CWRUDataError):
            slice_signal(np.arange(5, dtype=np.float32), window_size=0)
        with self.assertRaises(CWRUDataError):
            slice_signal(np.arange(5, dtype=np.float32), window_size=2, step=0)
        with self.assertRaises(CWRUDataError):
            load_cwru_signal(CWRU_ROOT / "raw" / "normal" / "Normal_0.txt")

    def test_nonfinite_signal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.mat"
            savemat(path, {"X001_DE_time": np.array([[1.0], [np.nan]])})
            with self.assertRaisesRegex(CWRUDataError, "NaN"):
                load_cwru_signal(path)


if __name__ == "__main__":
    unittest.main()

