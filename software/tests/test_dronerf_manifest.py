import tempfile
import unittest
from pathlib import Path

from airwatch.data.dronerf import (
    DroneRFArchiveEntry,
    DroneRFManifestError,
    DroneRFRecording,
    DroneRFSplitAssignment,
    assign_recording_splits,
    audit_recording_splits,
    build_recordings,
    decode_label,
    discover_recordings,
    parse_member_path,
    parse_package_path,
    read_recording_manifest,
    read_split_manifest,
    write_recording_manifest,
    write_split_manifest,
)


def entry(package, member, code, band, index):
    return DroneRFArchiveEntry(package, member, code, band, index)


class DroneRFManifestTests(unittest.TestCase):
    def test_decodes_documented_type_and_mode_labels(self):
        self.assertFalse(decode_label("00000").drone_present)
        self.assertEqual(decode_label("10011").operation_mode, "video_recording")
        self.assertEqual(decode_label("10110").drone_type, "parrot_ar")
        self.assertEqual(decode_label("11000").drone_type, "dji_phantom_3")
        with self.assertRaises(DroneRFManifestError):
            decode_label("99999")

    def test_parses_storage_chunks_and_official_background_typo(self):
        low = parse_package_path("DroneRF/Phantom drone/RF Data_11000_L2.rar")
        self.assertEqual((low.code, low.band, low.storage_part), ("11000", "L", 2))
        high = parse_package_path("Background RF activites/FR Data_00000_H2.rar")
        self.assertEqual((high.code, high.band, high.storage_part), ("00000", "H", 2))

    def test_member_must_agree_with_package(self):
        package = parse_package_path("RF Data_10000_L.rar")
        parsed = parse_member_path(package, "RF Data_10000_L/10000L_7.csv")
        self.assertEqual((parsed.code, parsed.band, parsed.segment_index), ("10000", "L", 7))
        self.assertIsNone(parse_member_path(package, "RF Data_10000_L/"))
        with self.assertRaises(DroneRFManifestError):
            parse_member_path(package, "10000H_7.csv")

    def test_pairs_phantom_by_segment_not_package_suffix(self):
        records = build_recordings([
            entry("RF Data_11000_L1.rar", "11000L_9.csv", "11000", "L", 9),
            entry("RF Data_11000_L2.rar", "11000L_10.csv", "11000", "L", 10),
            entry("RF Data_11000_H.rar", "11000H_9.csv", "11000", "H", 9),
            entry("RF Data_11000_H.rar", "11000H_10.csv", "11000", "H", 10),
        ])
        self.assertEqual([item.recording_id for item in records], [
            "dronerf-v1:11000:009", "dronerf-v1:11000:010"
        ])
        self.assertTrue(records[0].low_package_path.endswith("L1.rar"))
        self.assertTrue(records[1].low_package_path.endswith("L2.rar"))
        self.assertEqual(records[0].source_scenario_group, "dronerf-v1:11000")

    def test_rejects_missing_or_duplicate_band(self):
        low = entry("low.rar", "10000L_0.csv", "10000", "L", 0)
        with self.assertRaises(DroneRFManifestError):
            build_recordings([low])
        with self.assertRaises(DroneRFManifestError):
            build_recordings([low, low])

    def test_discovers_packages_and_writes_deterministic_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "RF Data_10000_L.rar"
            high = root / "RF Data_10000_H.rar"
            low.touch(); high.touch()
            members = {
                low.name: ["10000L_0.csv"],
                high.name: ["folder/10000H_0.csv", "folder"],
            }
            records = discover_recordings(root, member_lister=lambda path: members[path.name])
            destination = write_recording_manifest(root / "manifest.csv", records)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("recording_id,source_scenario_group", text)
            self.assertIn("dronerf-v1:10000:000", text)
            self.assertEqual(read_recording_manifest(destination), records)

    def test_split_is_stable_stratified_and_recording_disjoint(self):
        recordings = []
        for code in ("10000", "10100"):
            label = decode_label(code)
            for index in range(10):
                recording_id = f"dronerf-v1:{code}:{index:03d}"
                recordings.append(DroneRFRecording(
                    recording_id, f"dronerf-v1:{code}", code, True,
                    label.drone_type, label.operation_mode, index,
                    "low.rar", f"{code}L_{index}.csv",
                    "high.rar", f"{code}H_{index}.csv",
                ))
        first = assign_recording_splits(recordings, seed=20260908)
        second = assign_recording_splits(reversed(recordings), seed=20260908)
        self.assertEqual(first, second)
        report = audit_recording_splits(recordings, first)
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts_by_code"]["10000"], {
            "train": 6, "validation": 2, "test": 2
        })
        with tempfile.TemporaryDirectory() as directory:
            destination = write_split_manifest(Path(directory) / "split.csv", first)
            self.assertIn("split_seed", destination.read_text(encoding="utf-8"))
            self.assertEqual(read_split_manifest(destination), first)

    def test_split_audit_detects_duplicate_recording(self):
        recording = DroneRFRecording(
            "r1", "scenario", "10000", True, "parrot_bebop", "on_connected", 0,
            "low.rar", "low.csv", "high.rar", "high.csv",
        )
        assignment = DroneRFSplitAssignment("r1", "scenario", "10000", "train", 7)
        report = audit_recording_splits([recording], [assignment, assignment])
        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_recording_ids"], ["r1"])


if __name__ == "__main__":
    unittest.main()
