import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np

from airwatch.data.matlab_v73_iq_audit import (
    build_window_read_acceptance,
    write_window_read_acceptance,
)
from airwatch.data.matlab_v73_iq import (
    MATV73IQError,
    inspect_mat_v73_iq,
    inspect_mat_v73_iq_fileobj,
    read_mat_v73_iq_window,
)


class MATLABV73IQTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, *, shape=(1, 16), invalid_at=None) -> np.ndarray:
        dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
        values = np.zeros(16, dtype=dtype)
        values["real"] = np.arange(16, dtype=np.float64)
        values["imag"] = -np.arange(16, dtype=np.float64)
        if invalid_at is not None:
            values["imag"][invalid_at] = np.nan
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("uhd_samps", data=values.reshape(shape))
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
        return values

    def test_inspect_reads_metadata_without_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mat"
            self._write_iq(path)
            metadata = inspect_mat_v73_iq(path, sample_rate_hz=8.0)
            self.assertEqual(metadata.sample_count, 16)
            self.assertEqual(metadata.duration_seconds, 2.0)
            self.assertEqual(metadata.source_shape, (1, 16))
            self.assertEqual(metadata.matlab_class, "double")
            self.assertEqual(metadata.orientation, "row")

    def test_reads_only_requested_row_window_as_immutable_float32_iq(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mat"
            self._write_iq(path)
            window = read_mat_v73_iq_window(
                path, start_sample=3, window_size=4, sample_rate_hz=8.0
            )
            self.assertEqual(window.samples.shape, (2, 4))
            self.assertEqual(window.samples.dtype, np.float32)
            np.testing.assert_array_equal(window.samples[0], [3, 4, 5, 6])
            np.testing.assert_array_equal(window.samples[1], [-3, -4, -5, -6])
            self.assertFalse(window.samples.flags.writeable)
            self.assertEqual(window.end_sample, 7)
            self.assertEqual(window.duration_seconds, 0.5)

    def test_supports_singleton_column_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "column.mat"
            self._write_iq(path, shape=(16, 1))
            metadata = inspect_mat_v73_iq(path)
            window = read_mat_v73_iq_window(path, start_sample=14, window_size=2)
            self.assertEqual(metadata.orientation, "column")
            np.testing.assert_array_equal(window.samples[:, 1], [15, -15])

    def test_inspects_seekable_zip_member_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.mat"
            archive = root / "recordings.zip"
            self._write_iq(source)
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                handle.write(source, "group/recording.mat")
            with zipfile.ZipFile(archive, "r") as handle:
                with handle.open("group/recording.mat", "r") as member:
                    metadata = inspect_mat_v73_iq_fileobj(
                        member,
                        source_name="recordings.zip::group/recording.mat",
                        sample_rate_hz=8,
                    )
            self.assertEqual(metadata.sample_count, 16)
            self.assertEqual(metadata.path, "recordings.zip::group/recording.mat")

    def test_rejects_unbounded_or_out_of_range_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mat"
            self._write_iq(path)
            for invalid in (True, -1, 1.5):
                with self.subTest(start_sample=invalid):
                    with self.assertRaises(MATV73IQError):
                        read_mat_v73_iq_window(path, start_sample=invalid, window_size=1)
            with self.assertRaises(MATV73IQError):
                read_mat_v73_iq_window(path, start_sample=0, window_size=5, max_window_samples=4)
            with self.assertRaises(MATV73IQError):
                read_mat_v73_iq_window(path, start_sample=15, window_size=2)

    def test_rejects_missing_dataset_bad_shape_and_bad_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.mat"
            with h5py.File(missing, "w") as handle:
                handle.create_dataset("other", data=np.zeros(1))
            with self.assertRaises(MATV73IQError):
                inspect_mat_v73_iq(missing)

            bad_shape = root / "bad-shape.mat"
            dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
            with h5py.File(bad_shape, "w") as handle:
                handle.create_dataset("uhd_samps", data=np.zeros((2, 8), dtype=dtype))
            with self.assertRaises(MATV73IQError):
                inspect_mat_v73_iq(bad_shape)

            bad_dtype = root / "bad-dtype.mat"
            with h5py.File(bad_dtype, "w") as handle:
                handle.create_dataset("uhd_samps", data=np.zeros((1, 8), dtype=np.complex128))
            with self.assertRaises(MATV73IQError):
                inspect_mat_v73_iq(bad_dtype)

    def test_nonfinite_check_is_limited_to_requested_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mat"
            self._write_iq(path, invalid_at=12)
            valid = read_mat_v73_iq_window(path, start_sample=0, window_size=4)
            self.assertTrue(np.isfinite(valid.samples).all())
            with self.assertRaises(MATV73IQError):
                read_mat_v73_iq_window(path, start_sample=12, window_size=1)

    def test_acceptance_evidence_is_source_bound_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.mat"
            output = root / "acceptance.json"
            self._write_iq(source)
            report = build_window_read_acceptance(
                source, start_sample=4, window_size=4, sample_rate_hz=8.0
            )
            self.assertTrue(report["passed"])
            self.assertFalse(report["training_eligible"])
            self.assertEqual(report["bounded_read"]["output_shape"], [2, 4])
            self.assertEqual(len(report["source"]["sha256"]), 64)
            self.assertEqual(write_window_read_acceptance(report, output), output)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                write_window_read_acceptance(report, output)


if __name__ == "__main__":
    unittest.main()
