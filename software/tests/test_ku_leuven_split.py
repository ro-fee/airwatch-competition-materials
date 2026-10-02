import csv
import tempfile
import unittest
from pathlib import Path

from airwatch.data.ku_leuven_split import (
    KNOWN_ARCHIVES,
    KULeuvenKnownRecording,
    KULeuvenSplitError,
    assign_contiguous_proxy_splits,
    audit_known_splits,
    read_known_recordings,
    write_split_evidence,
)


class KULeuvenSplitTests(unittest.TestCase):
    @staticmethod
    def _recordings(count: int = 21):
        rows = []
        for archive_id in KNOWN_ARCHIVES:
            for index in range(count):
                rows.append(
                    KULeuvenKnownRecording(
                        recording_id=f"{archive_id}:{index}",
                        archive_id=archive_id,
                        member_index=index,
                        member_path=f"{archive_id}/{index}.mat",
                        device_group=archive_id,
                        device_label=archive_id,
                        member_sha256=f"{len(rows) + 1:064x}",
                    )
                )
        return tuple(rows)

    def test_contiguous_blocks_are_separated_and_deterministic(self):
        recordings = self._recordings()
        first = assign_contiguous_proxy_splits(recordings, guard_members=2)
        second = assign_contiguous_proxy_splits(reversed(recordings), guard_members=2)
        self.assertEqual(first, second)
        report = audit_known_splits(recordings, first, guard_members=2)
        self.assertTrue(report["ok"])
        self.assertFalse(report["physical_session_independence_established"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(
            report["counts_by_archive"]["frysky-v1"],
            {"train": 11, "validation": 3, "test": 3, "guard": 4},
        )

    def test_rejects_non_contiguous_members(self):
        recordings = list(self._recordings())
        recordings = [row for row in recordings if row.recording_id != "frysky-v1:5"]
        with self.assertRaises(KULeuvenSplitError):
            assign_contiguous_proxy_splits(recordings, guard_members=2)

    def test_evidence_binds_source_manifests_and_refuses_overwrite(self):
        recordings = self._recordings()
        assignments = assign_contiguous_proxy_splits(recordings, guard_members=2)
        audit = audit_known_splits(recordings, assignments, guard_members=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for index in range(3):
                path = root / f"manifest-{index}.csv"
                path.write_text("evidence\n", encoding="utf-8")
                manifests.append(path)
            outputs = write_split_evidence(assignments, audit, manifests, root / "split")
            with outputs["assignments"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 63)
            self.assertTrue(outputs["verification"].is_file())
            self.assertTrue(outputs["index"].is_file())
            with self.assertRaises(FileExistsError):
                write_split_evidence(assignments, audit, manifests, root / "split")


if __name__ == "__main__":
    unittest.main()
