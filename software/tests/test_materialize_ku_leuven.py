import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import h5py
import numpy as np

from airwatch.data.materialize_ku_leuven import (
    KULeuvenMaterializationError,
    materialize_known_windows,
)


class MaterializeKULeuvenTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _mat_bytes() -> bytes:
        stream = io.BytesIO()
        dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
        values = np.zeros((1, 5000), dtype=dtype)
        phase = np.linspace(0, 20 * np.pi, 5000)
        values["real"] = 2 + np.cos(phase)
        values["imag"] = -3 + np.sin(phase)
        with h5py.File(stream, "w") as handle:
            handle.create_dataset("uhd_samps", data=values)
        return stream.getvalue()

    def test_materializes_only_requested_train_windows_with_traceability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("group/sample.mat", self._mat_bytes())
            split_path = root / "split.csv"
            self._write_csv(split_path, [{
                "recording_id": "recording-0", "archive_id": "known-v1",
                "label_index": 2, "member_index": 0, "member_path": "group/sample.mat",
                "member_sha256": "a" * 64, "split": "train",
            }])
            plan_path = root / "plan.csv"
            self._write_csv(plan_path, [{
                "window_id": "recording-0:w000", "recording_id": "recording-0",
                "member_path": "group/sample.mat", "start_sample": 400,
                "end_sample_exclusive": 4496, "window_length": 4096,
            }])
            output = materialize_known_windows(
                split_assignments=split_path,
                sources=[("known-v1", archive, plan_path)],
                output_dir=root / "prepared",
                splits=("train",),
            )
            data = np.load(output / "train" / "data.npy")
            labels = np.load(output / "train" / "labels.npy")
            self.assertEqual(data.shape, (1, 2, 4096))
            self.assertEqual(labels.tolist(), [2])
            np.testing.assert_allclose(data.mean(axis=2), 0, atol=1e-5)
            rms = np.sqrt(np.mean(np.sum(data[0].astype(np.float64) ** 2, axis=0)))
            self.assertAlmostEqual(rms, 1, places=5)
            report = json.loads((output / "dataset-metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(report["training_data_ready"])
            self.assertFalse(report["test_materialized"])
            with self.assertRaises(FileExistsError):
                materialize_known_windows(
                    split_assignments=split_path,
                    sources=[("known-v1", archive, plan_path)],
                    output_dir=output,
                    splits=("train",),
                )

    def test_rejects_test_materialization(self):
        with self.assertRaises(KULeuvenMaterializationError):
            materialize_known_windows(
                split_assignments="missing.csv", sources=[], output_dir="unused", splits=("test",)
            )


if __name__ == "__main__":
    unittest.main()
