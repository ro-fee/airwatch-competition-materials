"""Integrity tests for the software-demo historical comparison manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airwatch.data.historical_comparison import (
    PERFORMANCE_STATUS,
    default_historical_comparison_manifest,
    prepare_historical_comparison,
)
from airwatch.general_recognition_contract import task_spec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_fixture(root: Path, *, samples_per_channel: int = 2048) -> Path:
    spec = task_spec("信号个体识别")
    signal = np.arange(2 * samples_per_channel, dtype=np.float32).reshape(
        samples_per_channel, 2
    )
    data_path = root / "recording.dat"
    signal.tofile(data_path)
    payload = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "task_name": "信号个体识别",
        "purpose": "software_demo",
        "performance_evidence": False,
        "origin": "unit test fixture",
        "label_provenance": "synthetic navigation metadata",
        "sample_dtype": spec.sample_dtype,
        "input_layout": spec.input_layout,
        "channels": spec.channels,
        "window_size": spec.window_size,
        "label_order": list(spec.classes),
        "recordings": [
            {
                "recording_id": "recording-1",
                "path": data_path.name,
                "sha256": _sha256(data_path),
                "bytes": data_path.stat().st_size,
                "label": spec.classes[0],
                "acquisition_group": "software-fixture",
            }
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return manifest


class HistoricalComparisonInputTests(unittest.TestCase):
    def test_bundled_manifest_is_verified_but_never_becomes_truth(self) -> None:
        prepared = prepare_historical_comparison(
            default_historical_comparison_manifest()
        )

        self.assertEqual(prepared.dataset_id, "historical-individual-software-demo-v1")
        self.assertEqual(prepared.recording_count, 4)
        self.assertEqual(prepared.windows_per_recording, (8, 8, 8, 8))
        self.assertEqual(prepared.total_windows, 32)
        self.assertFalse(prepared.performance_evidence_eligible)
        self.assertEqual(prepared.performance_status, PERFORMANCE_STATUS)
        self.assertEqual(prepared.snapshot().labels, ())
        self.assertTrue(all(not array.flags.writeable for array in prepared.inputs.arrays))

    def test_true_performance_evidence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_fixture(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["performance_evidence"] = True
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "performance_evidence=false"):
                prepare_historical_comparison(manifest)

    def test_hash_drift_is_refused_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_fixture(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["recordings"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                prepare_historical_comparison(manifest)

    def test_duplicate_identity_and_path_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_fixture(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["recordings"].append(dict(payload["recordings"][0]))
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "重复 recording_id"):
                prepare_historical_comparison(manifest)

    def test_path_traversal_and_contract_drift_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_fixture(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["recordings"][0]["path"] = "../recording.dat"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "清单目录内"):
                prepare_historical_comparison(manifest)

            manifest = _write_fixture(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["label_order"] = list(reversed(payload["label_order"]))
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "冻结历史任务标签顺序"):
                prepare_historical_comparison(manifest)

    def test_short_and_non_finite_inputs_are_refused(self) -> None:
        for samples, mutation, expected in (
            (100, None, "少于任务窗长"),
            (1024, "nan", "非有限值"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = _write_fixture(root, samples_per_channel=samples)
                if mutation == "nan":
                    values = np.fromfile(root / "recording.dat", dtype=np.float32)
                    values[0] = np.nan
                    values.tofile(root / "recording.dat")
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    payload["recordings"][0]["sha256"] = _sha256(root / "recording.dat")
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    prepare_historical_comparison(manifest)


if __name__ == "__main__":
    unittest.main()
