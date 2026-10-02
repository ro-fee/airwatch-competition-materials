"""Regression tests for the frozen anti-noise bearing model contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from training.verify_bearing_contract import (
    DEFAULT_CONTRACT,
    BearingContractError,
    verify_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / (
    "artifacts/checkpoints/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_best.pt"
)
LABEL_MAP = PROJECT_ROOT / "datasets/bearing/cwru/label-map.json"
QUALITY_CALIBRATION = PROJECT_ROOT / (
    "artifacts/evidence/bearing/bearing_quality_calibration_20260903_v2.json"
)


def _contract_digest(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("contract_digest_sha256", None)
    canonical = json.dumps(
        without_digest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


class FrozenBearingContractTests(unittest.TestCase):
    def test_current_frozen_contract_passes(self) -> None:
        result = verify_contract(DEFAULT_CONTRACT)

        self.assertTrue(result["passed"])
        self.assertEqual(result["contract_id"], "bearing_cnn_anti_noise_20260904_v1")
        self.assertEqual(result["checked_files"], 29)
        self.assertEqual(result["normalization"], "window_zscore")
        self.assertEqual(result["quality_gate_version"], "V2")

    def test_verification_does_not_modify_frozen_inputs(self) -> None:
        paths = (DEFAULT_CONTRACT, CHECKPOINT, LABEL_MAP, QUALITY_CALIBRATION)
        before = {
            path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
            for path in paths
        }

        verify_contract(DEFAULT_CONTRACT)

        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(
                    (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns),
                    before[path],
                )

    def test_contract_body_change_without_new_digest_is_rejected(self) -> None:
        original = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(original)
        changed["purpose"] = "tampered frozen contract"

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / DEFAULT_CONTRACT.name
            candidate.write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BearingContractError, "contract digest"):
                verify_contract(candidate)

    def test_changed_registered_hash_is_rejected_even_with_new_digest(self) -> None:
        original = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(original)
        checkpoint_record = next(
            record
            for record in changed["integrity"]["files"]
            if record["role"] == "checkpoint"
        )
        checkpoint_record["sha256"] = "0" * 64
        changed["contract_digest_sha256"] = _contract_digest(changed)

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / DEFAULT_CONTRACT.name
            candidate.write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BearingContractError, "SHA-256"):
                verify_contract(candidate)

    def test_rejected_state_cannot_enter_model_or_emit_prediction(self) -> None:
        contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        rejected = contract["quality_gate"]["states"]["rejected"]

        self.assertFalse(rejected["model_entry"])
        self.assertIsNone(rejected["prediction"])
        self.assertIsNone(rejected["confidence"])


if __name__ == "__main__":
    unittest.main()
