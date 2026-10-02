"""Tests for the read-only external bearing-model compatibility check."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from airwatch.data.external_compat import (
    build_cnn_transformer_compatibility_report,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/bearing/cwru/split-manifest.csv"
LABEL_MAP = ROOT / "datasets/bearing/cwru/label-map.json"


class ExternalBearingCompatibilityTests(unittest.TestCase):
    def test_report_uses_local_contract_and_refuses_direct_integration(self) -> None:
        report = build_cnn_transformer_compatibility_report(
            MANIFEST,
            LABEL_MAP,
            checked_at="2026-09-03T00:00:00+00:00",
        )

        self.assertEqual(report["decision"], "external_reference_only")
        self.assertFalse(report["can_directly_integrate"])
        self.assertFalse(report["weights_available"])
        self.assertFalse(report["trained"])
        self.assertFalse(report["ui_integrated"])
        self.assertFalse(report["historical_weights_modified"])
        self.assertEqual(report["local_contract"]["window_sizes"], [1024])
        self.assertEqual(report["local_contract"]["steps"], [1024])
        self.assertEqual(report["local_contract"]["window_counts"], {
            "train": 1422,
            "validation": 828,
            "test": 831,
        })
        self.assertEqual(report["checks"]["label_order"]["status"], "different")
        self.assertEqual(report["checks"]["weights"]["status"], "unavailable")
        self.assertEqual(report["checks"]["split"]["status"], "not_comparable_as_is")

    def test_report_is_json_serializable_and_write_isolated(self) -> None:
        report = build_cnn_transformer_compatibility_report(MANIFEST, LABEL_MAP)
        with tempfile.TemporaryDirectory() as directory:
            destination = write_report(report, Path(directory) / "compatibility.json")
            loaded = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(loaded["reference"]["name"], "CNN-Transformer-for-Bearing-Fault-Diagnosis")
        self.assertTrue(destination.name.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
