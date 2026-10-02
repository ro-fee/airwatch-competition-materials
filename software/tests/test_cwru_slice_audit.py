import csv
import json
import unittest
from pathlib import Path

from tools.audit_cwru_slices import LABEL_ORDER, audit, write_artifacts


ROOT = Path(__file__).resolve().parents[1]
CWRU_ROOT = ROOT / "datasets" / "bearing" / "cwru"


class TestCWRUSliceAudit(unittest.TestCase):
    def test_audit_is_file_grouped_and_balanced_by_class(self):
        result = audit(CWRU_ROOT)
        self.assertEqual(result["files_audited"], 16)
        self.assertEqual(result["total_windows"], 3081)
        self.assertEqual(result["label_order"], LABEL_ORDER)
        self.assertEqual(result["source_file_count_by_split"], {"test": 4, "train": 8, "validation": 4})
        for split in ("train", "validation", "test"):
            for class_name in LABEL_ORDER:
                self.assertEqual(result["source_file_count_by_class_and_split"][f"{split}:{class_name}"], 2 if split == "train" else 1)

    def test_normal_2_selection_is_explicit(self):
        result = audit(CWRU_ROOT)
        record = next(item for item in result["records"] if item["original_filename"] == "Normal_2.mat")
        self.assertEqual(record["sensor_key"], "X099_DE_time")
        self.assertEqual(result["policy"]["normal_2_sensor_choice"]["sensor_key"], "X099_DE_time")

    def test_outputs_are_reproducible_in_structure_and_have_no_processed_dir(self):
        slice_path, split_path, label_path = write_artifacts(CWRU_ROOT)
        self.assertTrue(slice_path.is_file())
        self.assertTrue(split_path.is_file())
        self.assertTrue(label_path.is_file())
        self.assertFalse((CWRU_ROOT / "processed").exists())

        data = json.loads(slice_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["records"]), 16)
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertEqual(sorted({row["split"] for row in rows}), ["test", "train", "validation"])


if __name__ == "__main__":
    unittest.main()

