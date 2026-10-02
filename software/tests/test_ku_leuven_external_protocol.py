import json
import tempfile
import unittest
from pathlib import Path

from airwatch.evidence.ku_leuven_external_protocol import (
    KULeuvenExternalProtocolError,
    validate_external_unknown_protocol,
    write_validation_report,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "training/configs/ku_leuven_sjrc_external_unknown_protocol_v2.json"
IN_DOMAIN_PROTOCOL = ROOT / "training/configs/ku_leuven_in_domain_known_unknown_protocol_v1.json"


class KULeuvenExternalProtocolTests(unittest.TestCase):
    def test_in_domain_protocol_keeps_paper_unknown_sources_evaluation_only(self):
        protocol = json.loads(IN_DOMAIN_PROTOCOL.read_text(encoding="utf-8"))
        known = {item["archive"] for item in protocol["known_sources"]}
        unknown = {item["archive"] for item in protocol["unknown_sources"]}
        self.assertFalse(known & unknown)
        self.assertEqual(
            unknown,
            {"NineEagles.zip", "wltoys.zip", "Q205.zip", "SJRC_pro.zip"},
        )
        self.assertTrue(all(item["evaluation_only"] for item in protocol["unknown_sources"]))
        rules = protocol["integrity_rules"]
        self.assertFalse(rules["unknown_training_allowed"])
        self.assertFalse(rules["unknown_threshold_selection_allowed"])
        self.assertTrue(protocol["evaluation"]["unknown_results_may_not_change_model_or_threshold"])

    def test_repository_protocol_is_source_bound_and_fail_closed(self):
        report = validate_external_unknown_protocol(PROTOCOL, project_root=ROOT)
        self.assertTrue(report["ok"])
        self.assertEqual(report["recording_count"], 51)
        self.assertEqual(report["window_count"], 1632)
        self.assertEqual(report["assigned_external_role"], "external_unknown_test")
        self.assertFalse(report["evaluation_run"])
        self.assertFalse(report["evaluation_eligible"])
        self.assertIn("real time-domain amplitude", report["blocker"])

    def test_rejects_permission_to_tune_on_external_unknown_data(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["role_assignment"]["threshold_selection_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(KULeuvenExternalProtocolError):
                validate_external_unknown_protocol(path, project_root=ROOT)

    def test_validation_report_refuses_overwrite(self):
        report = validate_external_unknown_protocol(PROTOCOL, project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            self.assertEqual(write_validation_report(report, path), path)
            with self.assertRaises(FileExistsError):
                write_validation_report(report, path)


if __name__ == "__main__":
    unittest.main()
