from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from airwatch.data.cwru_audit import (
    CWRUManifestAuditError,
    audit_manifest,
    write_audit_json,
)


ROOT = Path(__file__).resolve().parents[1]
CWRU_ROOT = ROOT / "datasets" / "bearing" / "cwru"


COLUMNS = [
    "dataset",
    "local_path",
    "original_filename",
    "class_name",
    "label_index",
    "load_hp",
    "rpm",
    "sample_rate_hz",
    "sensor_key",
    "window_size",
    "step",
    "window_count",
    "split",
    "split_group",
    "sha256",
    "source_url",
    "downloaded_at",
]


class TestCWRUManifestAudit(unittest.TestCase):
    def _make_workspace(
        self,
        *,
        signal: np.ndarray | None = None,
        sensor_key: str = "X001_DE_time",
        window_count: int = 3,
        window_size: int = 4,
        step: int = 2,
        split: str = "train",
        split_group: str = "load_hp_0",
        mat_name: str = "Normal_0.mat",
        mat_keys: dict[str, np.ndarray] | None = None,
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path]:
        holder = tempfile.TemporaryDirectory()
        root = Path(holder.name)
        mat_path = root / "datasets" / "bearing" / "cwru" / "raw" / "normal" / mat_name
        mat_path.parent.mkdir(parents=True)
        if mat_keys is None:
            if signal is None:
                signal = np.arange(8, dtype=np.float32)
            mat_keys = {"X001_DE_time": np.asarray(signal).reshape(-1, 1)}
        savemat(mat_path, mat_keys)

        relative_path = mat_path.relative_to(root).as_posix()
        digest = hashlib.sha256(mat_path.read_bytes()).hexdigest()
        manifest_path = root / "manifest.csv"
        row = {
            "dataset": "CWRU",
            "local_path": relative_path,
            "original_filename": mat_name,
            "class_name": "normal",
            "label_index": "0",
            "load_hp": "0",
            "rpm": "1796",
            "sample_rate_hz": "12000",
            "sensor_key": sensor_key,
            "window_size": str(window_size),
            "step": str(step),
            "window_count": str(window_count),
            "split": split,
            "split_group": split_group,
            "sha256": digest,
            "source_url": "https://example.invalid/cwru.mat",
            "downloaded_at": "2026-09-02T00:00:00+00:00",
        }
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(row)

        label_map_path = root / "label-map.json"
        label_map_path.write_text(
            json.dumps({"labels": {"0": "normal"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        return holder, root, manifest_path, label_map_path, mat_path

    def test_real_manifest_has_expected_inventory_and_only_explicit_channel_warning(self):
        report = audit_manifest(
            CWRU_ROOT / "split-manifest.csv",
            project_root=ROOT,
            label_map_path=CWRU_ROOT / "label-map.json",
            reference_manifest_path=CWRU_ROOT / "manifest.csv",
        )
        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["summary"]["row_count"], 16)
        self.assertEqual(report["summary"]["valid_rows"], 15)
        self.assertEqual(report["summary"]["error_count"], 0)
        self.assertEqual(report["summary"]["warning_count"], 1)
        self.assertEqual(report["summary"]["total_window_count"], 3081)
        self.assertEqual(report["summary"]["windows_by_split"], {"test": 831, "train": 1422, "validation": 828})
        self.assertEqual({issue["code"] for issue in report["issues"]}, {"multiple_de_channels"})
        self.assertFalse((CWRU_ROOT / "processed").exists())

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("dataset,local_path\nCWRU,file.mat\n", encoding="utf-8")
            with self.assertRaises(CWRUManifestAuditError):
                audit_manifest(manifest, project_root=directory)

    def test_missing_file_is_an_error(self):
        holder, root, manifest, labels, _ = self._make_workspace()
        self.addCleanup(holder.cleanup)
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("Normal_0.mat", "Missing.mat"),
            encoding="utf-8",
        )
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("file_missing", {issue["code"] for issue in report["issues"]})

    def test_sensor_key_must_exist_in_mat_file(self):
        holder, root, manifest, labels, _ = self._make_workspace(sensor_key="X999_DE_time")
        self.addCleanup(holder.cleanup)
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("signal_invalid", {issue["code"] for issue in report["issues"]})

    def test_ambiguous_sensor_key_is_rejected(self):
        holder, root, manifest, labels, _ = self._make_workspace(
            sensor_key="X001_DE_time|X002_DE_time",
            mat_keys={
                "X001_DE_time": np.arange(8, dtype=np.float32).reshape(-1, 1),
                "X002_DE_time": np.arange(8, dtype=np.float32).reshape(-1, 1),
            },
        )
        self.addCleanup(holder.cleanup)
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("ambiguous_sensor_key", {issue["code"] for issue in report["issues"]})

    def test_window_count_mismatch_is_an_error(self):
        holder, root, manifest, labels, _ = self._make_workspace(window_count=99)
        self.addCleanup(holder.cleanup)
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("window_count_mismatch", {issue["code"] for issue in report["issues"]})

    def test_sha256_mismatch_is_an_error(self):
        holder, root, manifest, labels, _ = self._make_workspace()
        self.addCleanup(holder.cleanup)
        text = manifest.read_text(encoding="utf-8").replace("," + hashlib.sha256((root / "datasets" / "bearing" / "cwru" / "raw" / "normal" / "Normal_0.mat").read_bytes()).hexdigest() + ",", ",deadbeef,")
        manifest.write_text(text, encoding="utf-8")
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("sha256_mismatch", {issue["code"] for issue in report["issues"]})

    def test_split_group_leakage_is_an_error(self):
        holder, root, manifest, labels, mat_path = self._make_workspace()
        self.addCleanup(holder.cleanup)
        second = root / "datasets" / "bearing" / "cwru" / "raw" / "normal" / "Normal_1.mat"
        second.write_bytes(mat_path.read_bytes())
        digest = hashlib.sha256(second.read_bytes()).hexdigest()
        with manifest.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writerow({
                "dataset": "CWRU", "local_path": second.relative_to(root).as_posix(),
                "original_filename": second.name, "class_name": "normal", "label_index": "0",
                "load_hp": "1", "rpm": "1772", "sample_rate_hz": "12000", "sensor_key": "X001_DE_time",
                "window_size": "4", "step": "2", "window_count": "3", "split": "validation",
                "split_group": "load_hp_0", "sha256": digest, "source_url": "https://example.invalid/1.mat",
                "downloaded_at": "2026-09-02T00:00:00+00:00",
            })
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("split_group_leakage", {issue["code"] for issue in report["issues"]})

    def test_nan_or_inf_signal_is_an_error(self):
        holder, root, manifest, labels, _ = self._make_workspace(
            signal=np.array([0, 1, np.nan, 3, 4, 5, 6, 7], dtype=np.float32)
        )
        self.addCleanup(holder.cleanup)
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "error")
        self.assertIn("signal_invalid", {issue["code"] for issue in report["issues"]})

    def test_valid_temp_mat_is_pass_and_source_bytes_are_unchanged(self):
        holder, root, manifest, labels, mat_path = self._make_workspace()
        self.addCleanup(holder.cleanup)
        before = mat_path.read_bytes()
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["total_window_count"], 3)
        self.assertEqual(mat_path.read_bytes(), before)

    def test_write_audit_json_is_atomic_and_parseable(self):
        holder, root, manifest, labels, _ = self._make_workspace()
        self.addCleanup(holder.cleanup)
        report = audit_manifest(manifest, project_root=root, label_map_path=labels)
        destination = root / "out" / "audit.json"
        written = write_audit_json(report, destination)
        self.assertEqual(written, destination.resolve())
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["status"], "pass")
        self.assertEqual(list(destination.parent.glob(".audit.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
