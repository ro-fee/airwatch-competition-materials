"""Traceability tests for the local KU Leuven desktop rehearsal export."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airwatch.data.data_provenance import sha256_file
from training.export_ku_leuven_runtime_demo import export_demo


DATASET_ROOT = Path(r"D:\AirWatch_Datasets\ku_leuven_drone_rf\prepared\known-iq-v1")


@unittest.skipUnless(DATASET_ROOT.is_dir(), "local KU Leuven prepared data is unavailable")
class KULeuvenRuntimeDemoExportTests(unittest.TestCase):
    def test_export_is_complete_traceable_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            result = export_demo(DATASET_ROOT, directory)
            array_path = Path(result["array"]["path"])
            metadata_path = Path(result["metadata_path"])

            windows = np.load(array_path, allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(windows.shape, (2, 32 * 4096))
            self.assertEqual(windows.dtype, np.float32)
            self.assertEqual(len(metadata["window_ids"]), 32)
            self.assertEqual(
                metadata["array"]["layout"], "channel_first_concatenated_windows"
            )
            self.assertEqual(metadata["array"]["window_count"], 32)
            self.assertEqual(metadata["array"]["window_samples"], 4096)
            self.assertEqual(metadata["source_split"], "validation")
            self.assertEqual(metadata["array"]["sha256"], sha256_file(array_path))
            self.assertFalse(metadata["test_data_used"])
            self.assertFalse(metadata["unknown_data_used"])
            self.assertFalse(metadata["competition_metric_claim_allowed"])
            self.assertFalse(metadata["redistribution_allowed_by_this_artifact"])

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                export_demo(DATASET_ROOT, directory)


if __name__ == "__main__":
    unittest.main()
