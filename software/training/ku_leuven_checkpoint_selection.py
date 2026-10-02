"""Frozen development-only checkpoint selection for KU Leuven worst-noise work."""

from __future__ import annotations

from math import isfinite
from statistics import fmean
from typing import Any, Mapping, Sequence

from training.common import TrainingConfigError


PROTOCOL_ID = "clean-guarded-minus5-worst-repeat-v1"
EXPECTED_REPEAT_SEEDS = (2026090901, 2026090902, 2026090903)
EXPECTED_SNR_DB = -5.0
EXPECTED_CLEAN_FLOOR = 0.97


def validate_checkpoint_selection(value: object) -> dict[str, Any] | None:
    """Validate the only stress-aware selection rule accepted by the trainer."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TrainingConfigError("checkpoint_selection 必须是 JSON 对象")
    if set(value) != {
        "protocol",
        "clean_recording_macro_f1_floor",
        "awgn_snr_db",
        "repeat_seeds",
    }:
        raise TrainingConfigError("checkpoint_selection 字段与冻结协议不一致")
    if value.get("protocol") != PROTOCOL_ID:
        raise TrainingConfigError("不支持的 KU Leuven 检查点选择协议")
    if value.get("clean_recording_macro_f1_floor") != EXPECTED_CLEAN_FLOOR:
        raise TrainingConfigError("clean 录制级 Macro-F1 守门线必须为 0.97")
    if value.get("awgn_snr_db") != EXPECTED_SNR_DB:
        raise TrainingConfigError("压力选择条件必须为 -5 dB AWGN")
    repeat_seeds = value.get("repeat_seeds")
    if (
        not isinstance(repeat_seeds, list)
        or tuple(repeat_seeds) != EXPECTED_REPEAT_SEEDS
    ):
        raise TrainingConfigError("压力选择必须使用冻结的三个重复种子")
    return {
        "protocol": PROTOCOL_ID,
        "clean_recording_macro_f1_floor": EXPECTED_CLEAN_FLOOR,
        "awgn_snr_db": EXPECTED_SNR_DB,
        "repeat_seeds": list(EXPECTED_REPEAT_SEEDS),
    }


def summarize_checkpoint_selection(
    clean_validation: Mapping[str, Any],
    stress_validations: Sequence[Mapping[str, Any]],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the auditable lexicographic selection values for one epoch."""
    normalized = validate_checkpoint_selection(definition)
    assert normalized is not None
    if len(stress_validations) != len(EXPECTED_REPEAT_SEEDS):
        raise TrainingConfigError("检查点选择缺少 -5 dB 重复验证结果")
    try:
        clean_score = float(clean_validation["recording"]["macro_f1"])
        clean_loss = float(clean_validation["loss"])
        stress_scores = [
            float(item["recording"]["macro_f1"])
            for item in stress_validations
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigError("检查点选择验证指标缺失") from exc
    values = [clean_score, clean_loss, *stress_scores]
    if not all(isfinite(item) for item in values):
        raise TrainingConfigError("检查点选择验证指标必须为有限数")
    eligible = clean_score >= normalized["clean_recording_macro_f1_floor"]
    rank = None
    if eligible:
        rank = [
            min(stress_scores),
            fmean(stress_scores),
            clean_score,
            -clean_loss,
        ]
    return {
        "protocol": PROTOCOL_ID,
        "eligible": eligible,
        "clean_recording_macro_f1_floor": normalized[
            "clean_recording_macro_f1_floor"
        ],
        "clean_recording_macro_f1": clean_score,
        "clean_window_loss": clean_loss,
        "stress_family": "awgn",
        "stress_snr_db": EXPECTED_SNR_DB,
        "repeat_seeds": list(EXPECTED_REPEAT_SEEDS),
        "repeat_recording_macro_f1": stress_scores,
        "worst_repeat_recording_macro_f1": min(stress_scores),
        "mean_repeat_recording_macro_f1": fmean(stress_scores),
        "selection_rank": rank,
    }


def is_checkpoint_selection_improvement(
    candidate: Mapping[str, Any], best: Mapping[str, Any] | None
) -> bool:
    """Compare two summaries using the frozen lexicographic order."""
    rank = candidate.get("selection_rank")
    if not candidate.get("eligible") or not isinstance(rank, list) or len(rank) != 4:
        return False
    if best is None:
        return True
    best_rank = best.get("selection_rank")
    if not isinstance(best_rank, list) or len(best_rank) != 4:
        raise TrainingConfigError("已保存的检查点选择排名无效")
    return tuple(float(value) for value in rank) > tuple(
        float(value) for value in best_rank
    )


def next_stale_epochs(
    stale_epochs: int, *, improved: bool, has_eligible_checkpoint: bool
) -> int:
    """Start early-stopping patience only after the first eligible checkpoint."""
    if isinstance(stale_epochs, bool) or stale_epochs < 0:
        raise ValueError("stale_epochs must be a non-negative integer")
    if improved:
        return 0
    if not has_eligible_checkpoint:
        return 0
    return stale_epochs + 1


SELECTION_RULE_DESCRIPTION = (
    "clean recording macro-F1 must be >=0.97; then maximize the minimum "
    "recording macro-F1 across three frozen -5 dB AWGN validation repeats; "
    "tie-break by repeat mean, clean recording macro-F1, then minimum clean "
    "validation window loss; test and unknown unopened"
)


__all__ = [
    "EXPECTED_CLEAN_FLOOR",
    "EXPECTED_REPEAT_SEEDS",
    "EXPECTED_SNR_DB",
    "PROTOCOL_ID",
    "SELECTION_RULE_DESCRIPTION",
    "is_checkpoint_selection_improvement",
    "next_stale_epochs",
    "summarize_checkpoint_selection",
    "validate_checkpoint_selection",
]
