"""Deployment smoke checks load models without reporting performance."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from airwatch.runtime_smoke import (
    run_runtime_model_smoke,
    write_runtime_model_smoke_report,
)


class RuntimeModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_runtime_model_smoke()

    def test_all_shipped_model_roles_load_and_forward_on_cpu(self) -> None:
        self.assertEqual(self.report["status"], "PASS")
        self.assertIn("not an accuracy evaluation", self.report["scope"])
        expected = {
            "historical:信号个体识别",
            "historical:信号调制识别",
            "historical:信号通联识别",
            "historical:信号业务识别",
            "historical:信号编码识别",
            "historical:generator",
            "historical:individual_reference",
            "bearing:diagnosis",
            "uav:known_source",
        }
        self.assertEqual(set(self.report["models"]), expected)
        self.assertEqual(
            self.report["models"]["uav:known_source"]["open_set_status"],
            "not_evaluated",
        )
        self.assertEqual(
            self.report["models"]["uav:known_source"]["workflow_window_count"],
            32,
        )
        self.assertTrue(
            self.report["models"]["uav:known_source"]["workflow_contract_id"]
        )
        self.assertEqual(
            self.report["models"]["uav:known_source"]["workflow_quality_status"],
            "caution",
        )
        self.assertEqual(
            self.report["models"]["historical:generator"]["output_shape"],
            [2, 1, 1024],
        )
        self.assertTrue(
            self.report["models"]["historical:generator"]["workflow_atomic_session"]
        )
        comparison = self.report["models"]["historical:individual_reference"]
        self.assertEqual(comparison["workflow_total_windows"], 32)
        self.assertTrue(comparison["workflow_atomic_evidence"])
        self.assertEqual(
            comparison["workflow_performance_status"],
            "not_eligible_software_demo",
        )
        self.assertFalse(comparison["workflow_accuracy_published"])
        self.assertEqual(comparison["workflow_evidence_schema"], 1)

    def test_report_writer_is_json_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime-smoke.json"
            report = write_runtime_model_smoke_report(output)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".json.tmp").exists())
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
