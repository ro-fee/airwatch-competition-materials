"""Run and summarize the frozen local KU Leuven three-seed comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from airwatch.data.data_provenance import sha256_file
from training.common import (
    TrainingConfigError,
    load_config,
    resolve_output_paths,
    resolve_project_path,
    write_json,
)
from training.ku_leuven_baseline import train_ku_leuven


EXPECTED_SEEDS = [20260909, 20260910, 20260911]
EXPECTED_MODELS = ["DroneRFTCN", "DroneRFResNet18"]


def load_matrix(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取本机复现实验矩阵：{exc}") from exc
    if payload.get("protocol") != "ku-leuven-known-source-local-reproducibility-v1":
        raise TrainingConfigError("本机复现实验矩阵协议不匹配")
    if payload.get("seeds") != EXPECTED_SEEDS:
        raise TrainingConfigError("本机复现实验必须使用冻结的三个种子")
    candidates = payload.get("candidates", [])
    if [item.get("model") for item in candidates] != EXPECTED_MODELS:
        raise TrainingConfigError("本机候选模型集合或顺序不匹配")
    comparison = payload.get("comparison", {})
    if comparison.get("test_data_allowed") is not False:
        raise TrainingConfigError("本机复现期间必须封存测试集")
    if comparison.get("unknown_data_allowed") is not False:
        raise TrainingConfigError("本机复现期间必须封存未知类")
    if comparison.get("formal_model_selection_claim_allowed") is not False:
        raise TrainingConfigError("本机复现不得声明正式模型选择结论")
    if payload.get("a800_protocol_replaced") is not False:
        raise TrainingConfigError("本机复现不得替代 A800 协议")
    payload["_matrix_path"] = str(source)
    return payload


def summarize_candidate_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["model"]), []).append(run)
    if set(grouped) != set(EXPECTED_MODELS):
        raise TrainingConfigError("本机复现证据必须同时包含 TCN 和 ResNet18")

    summaries = []
    for model in EXPECTED_MODELS:
        model_runs = sorted(grouped[model], key=lambda item: int(item["seed"]))
        seeds = [int(run["seed"]) for run in model_runs]
        if seeds != EXPECTED_SEEDS:
            raise TrainingConfigError(f"{model} 没有完整的三种子证据")
        scores = [float(run["validation_recording_macro_f1"]) for run in model_runs]
        if any(not isfinite(score) or score < 0.0 or score > 1.0 for score in scores):
            raise TrainingConfigError(f"{model} 的验证 Macro-F1 非法")
        parameter_counts = {int(run["parameter_count"]) for run in model_runs}
        if len(parameter_counts) != 1:
            raise TrainingConfigError(f"{model} 不同种子的参数量不一致")
        summaries.append(
            {
                "model": model,
                "seeds": seeds,
                "validation_recording_macro_f1": scores,
                "mean_validation_recording_macro_f1": fmean(scores),
                "standard_deviation_validation_recording_macro_f1": pstdev(scores),
                "minimum_validation_recording_macro_f1": min(scores),
                "maximum_validation_recording_macro_f1": max(scores),
                "parameter_count": next(iter(parameter_counts)),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["mean_validation_recording_macro_f1"],
            row["standard_deviation_validation_recording_macro_f1"],
            row["parameter_count"],
        ),
    )
    return summaries, ranked[0]


def _validate_training_evidence(
    payload: dict[str, Any],
    *,
    model: str,
    seed: int,
    run_name: str,
    evidence_path: Path,
) -> None:
    if (
        payload.get("status") != "completed_training_validation_only"
        or payload.get("run_name") != run_name
        or int(payload.get("seed", -1)) != seed
        or payload.get("model", {}).get("name") != model
        or payload.get("test_data_used") is not False
        or payload.get("unknown_data_used") is not False
        or payload.get("competition_test_claim_allowed") is not False
        or payload.get("validation_model_selection_evidence_allowed") is not True
    ):
        raise TrainingConfigError(f"训练证据不满足本机冻结协议：{evidence_path}")
    if not Path(str(payload.get("checkpoint_path", ""))).is_file():
        raise TrainingConfigError(f"训练证据对应的最佳检查点不存在：{evidence_path}")


def _run_record(
    payload: dict[str, Any], evidence_path: Path, *, reused: bool
) -> dict[str, Any]:
    return {
        "model": payload["model"]["name"],
        "seed": int(payload["seed"]),
        "run_name": payload["run_name"],
        "validation_recording_macro_f1": float(
            payload["best_validation_recording_macro_f1"]
        ),
        "validation_window_loss": float(payload["best_validation_window_loss"]),
        "completed_epochs": int(payload["completed_epochs"]),
        "parameter_count": int(payload["model"]["parameter_count"]),
        "checkpoint_path": payload["checkpoint_path"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "reused_existing_evidence": reused,
    }


def run_matrix(
    matrix: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    matrix_path = Path(matrix["_matrix_path"])
    output_path = resolve_project_path(project_root, matrix["comparison_output"])
    if not dry_run and output_path.exists():
        raise FileExistsError("拒绝覆盖已有本机三种子复现汇总证据")

    runs: list[dict[str, Any]] = []
    planned_runs: list[dict[str, Any]] = []
    common_identity: dict[str, Any] | None = None
    for candidate in matrix["candidates"]:
        config_path = resolve_project_path(project_root, candidate["config"])
        for seed in matrix["seeds"]:
            config = load_config(config_path)
            config["training"] = dict(config["training"])
            config["training"]["seed"] = int(seed)
            config["output"] = dict(config["output"])
            run_name = f"{candidate['run_prefix']}_seed{seed}_v1"
            config["output"]["run_name"] = run_name
            paths = resolve_output_paths(config, project_root, run_name)

            if dry_run:
                result = train_ku_leuven(
                    config,
                    dry_run=True,
                    run_name=run_name,
                    dataset_root_override=dataset_root,
                )
                planned_runs.append(
                    {
                        "model": candidate["model"],
                        "seed": seed,
                        "run_name": run_name,
                        "status": result["status"],
                        "training_evidence": result["planned_outputs"][
                            "training_evidence"
                        ],
                    }
                )
                continue

            evidence_path = paths["training_evidence"]
            reused = evidence_path.is_file()
            if reused:
                payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            else:
                payload = train_ku_leuven(
                    config,
                    run_name=run_name,
                    dataset_root_override=dataset_root,
                )
            _validate_training_evidence(
                payload,
                model=candidate["model"],
                seed=seed,
                run_name=run_name,
                evidence_path=evidence_path,
            )
            identity = payload.get("dataset", {}).get("data_identity")
            if common_identity is None:
                common_identity = identity
            elif identity != common_identity:
                raise TrainingConfigError("本机六次训练混用了不同数据身份")
            runs.append(_run_record(payload, evidence_path, reused=reused))

    if dry_run:
        return {
            "status": "dry_run",
            "matrix_path": str(matrix_path),
            "planned_runs": planned_runs,
            "test_data_used": False,
            "unknown_data_used": False,
        }

    summaries, leader = summarize_candidate_runs(runs)
    report = {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_local_validation_reproducibility",
        "status": "completed_development_validation_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "data_identity": common_identity,
        "environment_scope": "local_rtx_4060_laptop_gpu",
        "runs": runs,
        "candidate_summaries": summaries,
        "development_leader": leader,
        "comparison_rule": matrix["comparison"],
        "formal_model_selection_claim_allowed": False,
        "competition_test_claim_allowed": False,
        "a800_protocol_replaced": False,
        "test_data_used": False,
        "unknown_data_used": False,
        "limitations": [
            "Only 24 validation MAT members are available across three known classes.",
            "Official metadata does not prove that MAT members are independent physical acquisition sessions.",
            "This development comparison cannot be reported as competition test performance.",
        ],
        "next_step": "Use the local variance diagnosis to freeze one stability intervention before any sealed evaluation.",
    }
    write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_matrix(
        load_matrix(args.matrix),
        dataset_root=args.dataset_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_matrix", "run_matrix", "summarize_candidate_runs"]
