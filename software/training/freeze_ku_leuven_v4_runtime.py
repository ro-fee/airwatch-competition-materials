"""Freeze the accepted V4 KU Leuven candidate for desktop development use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from airwatch.data.data_provenance import sha256_file
from training.common import write_json


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/evidence/uav/ku_leuven/local_development"
ROBUSTNESS = EVIDENCE / "ku_leuven_tcn_weighted_worst_noise_validation_robustness_v4.json"
COMPARISON = EVIDENCE / "ku_leuven_weighted_worst_noise_local_comparison_v4.json"
PROTOCOL = ROOT / "training/configs/ku_leuven_weighted_worst_noise_local_protocol_v4.json"
CHECKPOINT = ROOT / (
    "artifacts/checkpoints/uav/ku_leuven/local_development/"
    "ku_leuven_tcn_weighted_worst_noise_seed20260910_v4_best.pt"
)
RESOLVED_CONFIG = EVIDENCE / "ku_leuven_tcn_weighted_worst_noise_seed20260910_v4_config.json"
DESTINATION = EVIDENCE / "ku_leuven_tcn_weighted_worst_noise_seed20260910_v4_runtime_contract.json"
SELECTED_PREDICTOR = "tcn_seed20260910"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _integrity_record(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _selection_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    sources = [source["id"] for source in evidence["sources"]]
    rows = {
        predictor_id: fmean(
            condition["metrics"][predictor_id]["recording"]["macro_f1"]
            for condition in evidence["conditions"]
        )
        for predictor_id in sources
    }
    if max(rows, key=rows.get) != SELECTED_PREDICTOR:
        raise ValueError("selected V4 runtime predictor is not the 40-condition leader")
    minus5 = {
        predictor_id: [
            condition["metrics"][predictor_id]["recording"]["macro_f1"]
            for condition in evidence["conditions"]
            if condition["family"] == "awgn" and condition["value"] == -5.0
        ]
        for predictor_id in sources
    }
    return {
        "method": "post_hoc_mean_recording_macro_f1_across_reused_40_condition_development_validation",
        "predictor_means": rows,
        "selected_predictor": SELECTED_PREDICTOR,
        "selected_minus5_repeat_macro_f1": minus5[SELECTED_PREDICTOR],
        "independent_model_selection_claim_allowed": False,
    }


def build_contract() -> dict[str, Any]:
    for path in (
        ROBUSTNESS,
        COMPARISON,
        PROTOCOL,
        CHECKPOINT,
        RESOLVED_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    evidence = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if (
        evidence.get("test_data_used") is not False
        or evidence.get("unknown_data_used") is not False
        or evidence.get("competition_test_claim_allowed") is not False
        or evidence.get("development_validation_only") is not True
        or comparison.get("acceptance_passed") is not True
        or comparison.get("test_data_used") is not False
        or comparison.get("unknown_data_used") is not False
    ):
        raise ValueError("V4 evidence is not an accepted development-only candidate")
    checkpoint_record = _integrity_record("checkpoint", CHECKPOINT)
    files = [
        checkpoint_record,
        _integrity_record("resolved_config", RESOLVED_CONFIG),
        _integrity_record("model_definition", ROOT / "airwatch/models/uav_baselines.py"),
        _integrity_record(
            "preprocessing_implementation",
            ROOT / "airwatch/data/ku_leuven_preprocessing.py",
        ),
        _integrity_record(
            "inference_implementation", ROOT / "airwatch/inference/ku_leuven.py"
        ),
        _integrity_record("development_protocol", PROTOCOL),
        _integrity_record("development_comparison", COMPARISON),
    ]
    body = {
        "schema_version": 1,
        "status": "development_frozen",
        "release_tier": "development_only",
        "contract_id": "ku-leuven-known-source-runtime-dev-v2",
        "created_on": "2026-09-09",
        "model": {
            "id": SELECTED_PREDICTOR,
            "version": "ku-leuven-tcn-weighted-worst-noise-dev-20260910-v4",
            "name": "DroneRFTCN",
            "training_seed": 20260910,
            "checkpoint": checkpoint_record["path"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "resolved_config": _relative(RESOLVED_CONFIG),
        },
        "input_contract": {
            "format": "float IQ [32,2,4096], float IQ [2,131072], or complex [131072]",
            "preprocessing_id": "ku-leuven-iq-dc-rms-v1",
            "sample_rate_hz": 100000000,
            "channels": 2,
            "channel_semantics": ["in_phase", "quadrature"],
            "window_samples": 4096,
            "label_map": {
                "frysky": 0,
                "spektrum_dx4e": 1,
                "dji_mini2_rc": 2,
            },
            "display_labels": {
                "frysky": "FrSky 遥控信号",
                "spektrum_dx4e": "Spektrum DX4e 遥控信号",
                "dji_mini2_rc": "DJI Mini 2 遥控信号",
            },
        },
        "aggregation": {
            "protocol": "mean-window-softmax-probability-v1",
            "windows_per_recording": 32,
        },
        "open_set": {
            "status": "not_evaluated",
            "threshold": None,
            "known_unknown_publication_allowed": False,
            "message": "No frozen KU Leuven unknown-rejection threshold is available; low confidence must not be relabeled as unknown.",
        },
        "development_selection": {
            "source_evidence": _relative(ROBUSTNESS),
            "source_evidence_sha256": sha256_file(ROBUSTNESS),
            "accepted_comparison": _relative(COMPARISON),
            "accepted_comparison_sha256": sha256_file(COMPARISON),
            **_selection_summary(evidence),
        },
        "claims": {
            "known_source_classification_available": True,
            "unknown_rejection_available": False,
            "drone_presence_detection_available": False,
            "operation_state_classification_available": False,
            "competition_test_claim_allowed": False,
            "test_data_used": False,
            "unknown_data_used": False,
        },
        "limitations": [
            "Development-only model selected after repeated use of 24 known validation MAT members and 40 stress conditions.",
            "Labels describe three known RF control sources, not a general drone-presence detector.",
            "Open-set rejection, operation state, independent test performance, and cross-source generalization are not available.",
            "The -5 dB improvement is restricted to reused synthetic development stressors and is not independent or physical-SNR evidence.",
        ],
        "integrity": {"algorithm": "sha256", "files": files},
    }
    return {
        **body,
        "contract_digest_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    contract = build_contract()
    write_json(DESTINATION, contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
