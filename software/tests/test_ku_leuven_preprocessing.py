import io
import unittest

import h5py
import numpy as np

from airwatch.data.ku_leuven_preprocessing import (
    INPUT_WINDOW_SAMPLES,
    KULeuvenPreprocessingError,
    preprocess_iq_window,
)
from airwatch.data.matlab_v73_iq import (
    MATV73IQError,
    read_mat_v73_iq_window_fileobj,
    read_mat_v73_iq_windows_fileobj,
)


class KULeuvenPreprocessingTests(unittest.TestCase):
    def test_removes_dc_normalizes_complex_rms_and_preserves_input(self):
        phase = np.linspace(0, 20 * np.pi, INPUT_WINDOW_SAMPLES, dtype=np.float32)
        source = np.stack((3 + 4 * np.cos(phase), -2 + 4 * np.sin(phase)))
        before = source.copy()
        output = preprocess_iq_window(source)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(output.mean(axis=1), 0, atol=1e-6)
        rms = np.sqrt(np.mean(np.sum(output.astype(np.float64) ** 2, axis=0)))
        self.assertAlmostEqual(rms, 1.0, places=6)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(output.flags.writeable)

    def test_is_invariant_to_common_scale_and_per_channel_dc(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(2, INPUT_WINDOW_SAMPLES)).astype(np.float32)
        shifted = source * 3.5 + np.array([[7], [-4]], dtype=np.float32)
        np.testing.assert_allclose(
            preprocess_iq_window(source), preprocess_iq_window(shifted), atol=2e-6
        )

    def test_rejects_wrong_shape_nonfinite_and_zero_energy(self):
        with self.assertRaises(KULeuvenPreprocessingError):
            preprocess_iq_window(np.zeros((2, 16), dtype=np.float32))
        nonfinite = np.zeros((2, INPUT_WINDOW_SAMPLES), dtype=np.float32)
        nonfinite[0, 0] = np.nan
        with self.assertRaises(KULeuvenPreprocessingError):
            preprocess_iq_window(nonfinite)
        with self.assertRaises(KULeuvenPreprocessingError):
            preprocess_iq_window(np.ones((2, INPUT_WINDOW_SAMPLES), dtype=np.float32))

    def test_reads_bounded_window_from_seekable_mat_stream(self):
        stream = io.BytesIO()
        dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
        values = np.zeros((1, 5000), dtype=dtype)
        values["real"] = np.arange(5000)
        values["imag"] = -np.arange(5000)
        with h5py.File(stream, "w") as handle:
            dataset = handle.create_dataset("uhd_samps", data=values)
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
        stream.seek(0)
        window = read_mat_v73_iq_window_fileobj(
            stream, source_name="archive.zip::sample.mat", start_sample=100, window_size=4096
        )
        self.assertEqual(window.samples.shape, (2, 4096))
        self.assertEqual(window.samples[0, 0], 100)
        self.assertEqual(window.samples[1, -1], -4195)
        self.assertFalse(window.samples.flags.writeable)

    def test_reads_multiple_ordered_windows_with_one_stream(self):
        stream = io.BytesIO()
        dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
        values = np.zeros((1, 9000), dtype=dtype)
        values["real"] = np.arange(9000)
        values["imag"] = -np.arange(9000)
        with h5py.File(stream, "w") as handle:
            handle.create_dataset("uhd_samps", data=values)
        stream.seek(0)
        windows = read_mat_v73_iq_windows_fileobj(
            stream,
            source_name="archive.zip::sample.mat",
            start_samples=(0, 4096),
            window_size=4096,
        )
        self.assertEqual([window.start_sample for window in windows], [0, 4096])
        self.assertEqual(windows[1].samples[0, 0], 4096)
        stream.seek(0)
        with self.assertRaises(MATV73IQError):
            read_mat_v73_iq_windows_fileobj(
                stream,
                source_name="archive.zip::sample.mat",
                start_samples=(4096, 0),
                window_size=4096,
            )


if __name__ == "__main__":
    unittest.main()
