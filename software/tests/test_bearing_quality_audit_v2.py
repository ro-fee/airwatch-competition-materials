from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.audit_bearing_quality_v2 import (
    build_raw_test_dataset,
    ensure_outputs_absent,
    output_paths,
    validate_quality_prediction_rows_v2,
)


class BearingQualityAuditV2Test(unittest.TestCase):
    def test_dataset_contract_is_fixed_test_and_raw(self) -> None:
        config = {
            "dataset": {
                "manifest": "datasets/bearing/cwru/split-manifest.csv",
                "label_map": "datasets/bearing/cwru/label-map.json",
            }
        }
        fake_dataset = object()
        with mock.patch(
            "training.audit_bearing_quality_v2.CWRUBearingDataset",
            return_value=fake_dataset,
        ) as constructor:
            result = build_raw_test_dataset(config, Path.cwd())

        self.assertIs(result, fake_dataset)
        self.assertEqual(constructor.call_args.kwargs["split"], "test")
        self.assertEqual(constructor.call_args.kwargs["normalization"], "none")

    def test_existing_v2_evidence_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = output_paths(Path(temp_dir), "audit_v2")
            paths["json"].parent.mkdir(parents=True, exist_ok=True)
            paths["json"].write_text("historical evidence", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                ensure_outputs_absent(paths)

            self.assertEqual(
                paths["json"].read_text(encoding="utf-8"),
                "historical evidence",
            )

    def test_v2_status_controls_model_entry_contract(self) -> None:
        validate_quality_prediction_rows_v2(
            [
                {"status": "accepted", "accepted": True, "prediction": 0, "confidence": 0.9},
                {"status": "caution", "accepted": True, "prediction": 1, "confidence": 0.6},
                {"status": "rejected", "accepted": False, "prediction": None, "confidence": None},
            ]
        )
        with self.assertRaisesRegex(ValueError, "bypass the model"):
            validate_quality_prediction_rows_v2(
                [{"status": "rejected", "accepted": False, "prediction": 0, "confidence": 0.9}]
            )
        with self.assertRaisesRegex(ValueError, "must enter the model"):
            validate_quality_prediction_rows_v2(
                [{"status": "caution", "accepted": False, "prediction": None, "confidence": None}]
            )


if __name__ == "__main__":
    unittest.main()
