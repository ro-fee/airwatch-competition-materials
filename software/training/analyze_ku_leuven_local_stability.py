"""Create a post-hoc stability diagnostic from local KU Leuven run evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import isclose
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from airwatch.data.data_provenance import sha256_file
from training.common import TrainingConfigError, write_json


def summarize_run_history(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise TrainingConfigError("训练证据缺少非空 history")
    best_score = float(payload["best_validation_recording_macro_f1"])
    candidates = [
        item
        for item in history
        if float(item["validation"]["recording"]["macro_f1"]) == best_score
    ]
    if not candidates:
        raise TrainingConfigError("history 无法复现记录级最优检查点")
    selected = min(candidates, key=lambda item: float(item["validation"]["loss"]))
    selected_loss = float(selected["validation"]["loss"])
    if not isclose(
        selected_loss,
        float(payload["best_validation_window_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingConfigError("history 与训练证据记录的最优验证损失不一致")
    trajectory = [
        float(item["validation"]["recording"]["macro_f1"]) for item in history
    ]
    return {
        "model": payload["model"]["name"],
        "seed": int(payload["seed"]),
        "run_name": payload["run_name"],
        "selected_checkpoint_epoch": int(selected["epoch"]),
        "selected_checkpoint_recording_macro_f1": best_score,
        "selected_checkpoint_window_macro_f1": float(
            selected["validation"]["window"]["macro_f1"]
        ),
        "selected_checkpoint_window_accuracy": float(
            selected["validation"]["window"]["accuracy"]
        ),
        "selected_checkpoint_validation_loss": selected_loss,
        "selected_checkpoint_recording_per_class_recall": {
            name: float(metrics["recall"])
            for name, metrics in selected["validation"]["recording"][
                "per_class"
            ].items()
        },
        "trajectory_recording_macro_f1_mean": fmean(trajectory),
        "trajectory_recording_macro_f1_standard_deviation": pstdev(trajectory),
        "trajectory_recording_macro_f1_minimum": min(trajectory),
        "trajectory_recording_macro_f1_maximum": max(trajectory),
        "perfect_recording_macro_f1_epoch_count": sum(
            value == 1.0 for value in trajectory
        ),
        "completed_epochs": len(history),
    }


def _candidate_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in ("DroneRFTCN", "DroneRFResNet18"):
        selected = [item for item in runs if item["model"] == model]
        if sorted(item["seed"] for item in selected) != [20260909, 20260910, 20260911]:
            raise TrainingConfigError(f"{model} 的三种子稳定性证据不完整")
        window_scores = [
            item["selected_checkpoint_window_macro_f1"] for item in selected
        ]
        losses = [item["selected_checkpoint_validation_loss"] for item in selected]
        trajectory_std = [
            item["trajectory_recording_macro_f1_standard_deviation"]
            for item in selected
        ]
        rows.append(
            {
                "model": model,
                "selected_checkpoint_window_macro_f1": window_scores,
                "mean_selected_checkpoint_window_macro_f1": fmean(window_scores),
                "standard_deviation_selected_checkpoint_window_macro_f1": pstdev(
                    window_scores
                ),
                "minimum_selected_checkpoint_window_macro_f1": min(window_scores),
                "mean_selected_checkpoint_validation_loss": fmean(losses),
                "mean_within_run_recording_macro_f1_standard_deviation": fmean(
                    trajectory_std
                ),
            }
        )
    return rows


def build_stability_diagnostic(
    comparison_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    source = Path(comparison_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError("拒绝覆盖已有本机稳定性诊断证据")
    try:
        comparison = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取本机三种子汇总证据：{exc}") from exc
    if (
        comparison.get("artifact_type")
        != "ku_leuven_local_validation_reproducibility"
        or comparison.get("status") != "completed_development_validation_only"
        or comparison.get("test_data_used") is not False
        or comparison.get("unknown_data_used") is not False
        or comparison.get("formal_model_selection_claim_allowed") is not False
        or comparison.get("a800_protocol_replaced") is not False
    ):
        raise TrainingConfigError("本机三种子汇总证据不满足稳定性诊断边界")

    run_summaries = []
    for run in comparison.get("runs", []):
        evidence_path = Path(run["evidence_path"])
        if sha256_file(evidence_path) != run["evidence_sha256"]:
            raise TrainingConfigError(f"训练证据哈希不匹配：{evidence_path}")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            payload.get("run_name") != run["run_name"]
            or payload.get("test_data_used") is not False
            or payload.get("unknown_data_used") is not False
        ):
            raise TrainingConfigError(f"训练证据身份或数据边界不匹配：{evidence_path}")
        summary = summarize_run_history(payload)
        summary["evidence_path"] = str(evidence_path)
        summary["evidence_sha256"] = run["evidence_sha256"]
        run_summaries.append(summary)
    if len(run_summaries) != 6:
        raise TrainingConfigError("稳定性诊断需要完整的六次训练证据")

    candidate_summaries = _candidate_summaries(run_summaries)
    diagnostic_leader = sorted(
        candidate_summaries,
        key=lambda item: (
            -item["mean_selected_checkpoint_window_macro_f1"],
            item["mean_selected_checkpoint_validation_loss"],
        ),
    )[0]
    report = {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_local_training_stability_diagnostic",
        "status": "completed_post_hoc_development_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_comparison_path": str(source),
        "source_comparison_sha256": sha256_file(source),
        "runs": run_summaries,
        "candidate_summaries": candidate_summaries,
        "post_hoc_diagnostic_leader": diagnostic_leader,
        "post_hoc_diagnostic_only": True,
        "formal_model_selection_claim_allowed": False,
        "competition_test_claim_allowed": False,
        "test_data_used": False,
        "unknown_data_used": False,
        "interpretation_boundary": (
            "Recording-level validation saturated at 1.0 for every selected checkpoint; "
            "window metrics and epoch trajectories are reported only to diagnose the next frozen development protocol."
        ),
    }
    write_json(destination, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_stability_diagnostic(args.comparison, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_stability_diagnostic", "summarize_run_history"]
