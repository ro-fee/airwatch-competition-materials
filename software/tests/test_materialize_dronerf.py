import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from airwatch.data.dronerf import (
    DroneRFRecording,
    DroneRFSplitAssignment,
    write_recording_manifest,
    write_split_manifest,
)
from airwatch.data.dronerf_windows import DroneRFWindowResult, DroneRFWindowSpec
from airwatch.data.materialize_dronerf import materialize, verify_materialized_dataset


class DroneRFMaterializeTests(unittest.TestCase):
    def test_materializes_atomic_indexed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = []
            assignments = []
            for index, split in enumerate(("train", "validation", "test")):
                recording_id = f"dronerf-v1:10000:{index:03d}"
                recordings.append(DroneRFRecording(
                    recording_id, "dronerf-v1:10000", "10000", True,
                    "parrot_bebop", "on_connected", index,
                    "low.rar", f"10000L_{index}.csv",
                    "high.rar", f"10000H_{index}.csv",
                ))
                assignments.append(DroneRFSplitAssignment(
                    recording_id, "dronerf-v1:10000", "10000", split, 7
                ))
            recording_path = write_recording_manifest(root / "recordings.csv", recordings)
            split_path = write_split_manifest(root / "splits.csv", assignments)
            package_root = root / "packages"; package_root.mkdir()
            output = root / "processed"
            spec = DroneRFWindowSpec(window_length=16, windows_per_recording=2,
                                     expected_sample_count=64, seed=7)

            def fake_extract(recording, unused_root, unused_spec):
                samples = np.full((2, 2, 16), recording.segment_index, dtype=np.float32)
                return DroneRFWindowResult(samples, (0, 32), 64, 64)

            with patch(
                "airwatch.data.materialize_dronerf.extract_recording_windows",
                side_effect=fake_extract,
            ):
                metadata = materialize(
                    recording_path, split_path, package_root, output, spec
                )
            self.assertEqual(metadata["recording_count"], 3)
            self.assertEqual(metadata["window_count"], 6)
            self.assertEqual(metadata["counts_by_split"], {
                "test": 2, "train": 2, "validation": 2
            })
            self.assertFalse(output.with_name("processed.partial").exists())
            self.assertEqual(len(list((output / "recordings").rglob("*.npy"))), 3)
            with (output / "window-manifest.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["preprocessing_version"], spec.preprocessing_version)
            loaded = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["normalization"], "none; raw float32 RF amplitude windows")
            verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
            self.assertTrue(verification["ok"])
            self.assertEqual(verification["verified_window_rows"], 6)
            self.assertTrue(verify_materialized_dataset(
                output, recording_manifest=recording_path, split_manifest=split_path
            )["ok"])

if __name__ == "__main__":
    unittest.main()
