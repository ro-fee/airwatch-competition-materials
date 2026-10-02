"""Select the KU Leuven model from the complete frozen A800 validation matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from airwatch.data.data_provenance import sha256_file
from training.common import TrainingConfigError, load_config, resolve_project_path, write_json
from training.run_ku_leuven_a800_matrix import load_matrix


def select_candidate_summaries(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["model"]), []).append(run)
    if set(grouped) != {"DroneRFTCN", "DroneRFResNet18"}:
        raise TrainingConfigError("选模证据必须同时包含 TCN 和 ResNet18")
    summaries = []
    for model in ("DroneRFTCN", "DroneRFResNet18"):
        model_runs = grouped[model]
        seeds = sorted(int(run["seed"]) for run in model_runs)
        if seeds != [20260909, 20260910, 20260911]:
            raise TrainingConfigError(f"{model} 没有完整的三种子证据")
        scores = np.asarray(
            [float(run["validation_recording_macro_f1"]) for run in model_runs],
            dtype=np.float64,
        )
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise TrainingConfigError(f"{model} 的验证 Macro-F1 非法")
        parameter_counts = {int(run["parameter_count"]) for run in model_runs}
        if len(parameter_counts) != 1:
            raise TrainingConfigError(f"{model} 不同种子的参数量不一致")
        summaries.append({
            "model": model,
            "seeds": seeds,
            "validation_recording_macro_f1": [float(value) for value in scores],
            "mean_validation_recording_macro_f1": float(np.mean(scores)),
            "standard_deviation_validation_recording_macro_f1": float(np.std(scores)),
            "minimum_validation_recording_macro_f1": float(np.min(scores)),
            "maximum_validation_recording_macro_f1": float(np.max(scores)),
            "parameter_count": next(iter(parameter_counts)),
        })
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["mean_validation_recording_macro_f1"],
            row["standard_deviation_validation_recording_macro_f1"],
            row["parameter_count"],
        ),
    )
    return summaries, ranked[0]


def select_from_matrix(matrix_path: str | Path) -> dict[str, Any]:
    matrix_file = Path(matrix_path).resolve()
    matrix = load_matrix(matrix_file)
    project_root = Path(__file__).resolve().parents[1]
    output = resolve_project_path(project_root, matrix["selection_output"])
    if output.exists():
        raise FileExistsError("拒绝覆盖已有 A800 选模证据")
    runs = []
    common_identity = None
    for candidate in matrix["candidates"]:
        base_config = load_config(resolve_project_path(project_root, candidate["config"]))
        evidence_dir = resolve_project_path(project_root, base_config["output"]["evidence_dir"])
        for seed in matrix["seeds"]:
            run_name = f"{candidate['run_prefix']}_seed{seed}_v1"
            evidence_path = evidence_dir / f"{run_name}_training.json"
            if not evidence_path.is_file():
                raise FileNotFoundError(f"六次训练证据不完整：{evidence_path}")
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (
                payload.get("status") != "completed_training_validation_only"
                or payload.get("run_name") != run_name
                or int(payload.get("seed", -1)) != seed
                or payload.get("model", {}).get("name") != candidate["model"]
                or payload.get("test_data_used") is not False
                or payload.get("unknown_data_used") is not False
                or payload.get("competition_test_claim_allowed") is not False
                or payload.get("validation_model_selection_evidence_allowed") is not True
            ):
                raise TrainingConfigError(f"训练证据不满足冻结选模协议：{evidence_path}")
            identity = payload.get("dataset", {}).get("data_identity")
            if common_identity is None:
                common_identity = identity
            elif identity != common_identity:
                raise TrainingConfigError("六次训练混用了不同数据身份")
            runs.append({
                "model": candidate["model"],
                "seed": seed,
                "run_name": run_name,
                "validation_recording_macro_f1": payload["best_validation_recording_macro_f1"],
                "parameter_count": payload["model"]["parameter_count"],
                "checkpoint_path": payload["checkpoint_path"],
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
            })
    summaries, winner = select_candidate_summaries(runs)
    report = {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_a800_validation_model_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_file),
        "matrix_sha256": sha256_file(matrix_file),
        "data_identity": common_identity,
        "runs": runs,
        "candidate_summaries": summaries,
        "winner": winner,
        "selection_rule": matrix["selection"],
        "test_data_used": False,
        "unknown_data_used": False,
        "next_step": "Calibrate the selected checkpoint on known validation recordings only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args(argv)
    report = select_from_matrix(args.matrix)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select_candidate_summaries", "select_from_matrix"]
