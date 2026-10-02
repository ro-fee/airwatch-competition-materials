"""Run the V3 clean-guarded worst--5-dB checkpoint-selection experiment."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from airwatch.data.data_provenance import sha256_file
from training.common import TrainingConfigError, load_config, write_json
from training.evaluate_ku_leuven_robustness import run as evaluate
from training.ku_leuven_baseline import train_ku_leuven
from training.ku_leuven_checkpoint_selection import validate_checkpoint_selection


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/evidence/uav/ku_leuven/local_development"
PROTOCOL_PATH = (
    ROOT
    / "training/configs/ku_leuven_worst_noise_selection_local_protocol_v3.json"
)
PRIOR_EVIDENCE_PATH = (
    EVIDENCE / "ku_leuven_tcn_strong_noise_validation_robustness_v2.json"
)
COMPARISON_PATH = (
    EVIDENCE / "ku_leuven_worst_noise_selection_local_comparison_v3.json"
)
EXPECTED_PROTOCOL = "ku-leuven-worst-noise-selection-local-development-v3"
EXPECTED_SEEDS = [20260909, 20260910, 20260911]


def load_v3_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取 V3 协议：{source}: {exc}") from exc
    if not isinstance(protocol, dict) or protocol.get("protocol") != EXPECTED_PROTOCOL:
        raise TrainingConfigError("V3 协议标识不匹配")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise TrainingConfigError("V3 必须使用冻结的三个随机种子")
    if protocol.get("augmentation") != {
        "protocol": "ku-leuven-clean-strong-awgn-multipath-training-v2"
    }:
        raise TrainingConfigError("V3 训练增强必须与 V2 一致")
    validate_checkpoint_selection(protocol.get("checkpoint_selection"))
    required_true = ("development_validation_only",)
    required_false = (
        "test_data_allowed",
        "unknown_data_allowed",
        "open_set_threshold_calibration_allowed",
        "competition_test_claim_allowed",
        "a800_protocol_replaced",
    )
    if any(protocol.get(name) is not True for name in required_true) or any(
        protocol.get(name) is not False for name in required_false
    ):
        raise TrainingConfigError("V3 数据封存或声明边界不满足要求")
    references = protocol.get("reference_evidence")
    if not isinstance(references, dict):
        raise TrainingConfigError("V3 缺少参考证据")
    for name in ("base_config", "v2_protocol", "v2_robustness", "v2_comparison"):
        relative = references.get(name)
        expected = references.get(name + "_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TrainingConfigError(f"V3 参考证据 {name} 不完整")
        target = (ROOT / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise TrainingConfigError(f"V3 参考证据哈希不匹配：{relative}")
    protocol["_protocol_path"] = str(source)
    return protocol


def _mean_group(
    rows: list[dict[str, Any]],
    member_ids: list[str],
    selector: Callable[[dict[str, Any]], bool],
) -> dict[str, float]:
    selected = [
        row for row in rows if row["predictor"] in member_ids and selector(row)
    ]
    if not selected:
        raise ValueError("comparison group is empty")
    return {
        key: fmean(float(row[key]) for row in selected)
        for key in ("baseline", "candidate", "difference")
    }


def compare(prior: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Make the pre-registered exactly paired V2-versus-V3 comparison."""
    if (
        [item["condition_id"] for item in prior["conditions"]]
        != [item["condition_id"] for item in candidate["conditions"]]
        or prior["dataset"] != candidate["dataset"]
    ):
        raise ValueError("paired comparison requires identical conditions and data")
    member_ids = [source["id"] for source in prior["sources"]]
    if member_ids != [source["id"] for source in candidate["sources"]]:
        raise ValueError("paired comparison requires identical predictor identities")
    ensemble_id = prior["ensemble"]["id"]
    if ensemble_id != candidate["ensemble"]["id"]:
        raise ValueError("paired comparison requires identical ensemble identity")

    rows: list[dict[str, Any]] = []
    for before, after in zip(prior["conditions"], candidate["conditions"]):
        for predictor in member_ids + [ensemble_id]:
            baseline = before["metrics"][predictor]["recording"]["macro_f1"]
            result = after["metrics"][predictor]["recording"]["macro_f1"]
            rows.append(
                {
                    "condition_id": before["condition_id"],
                    "family": before["family"],
                    "value": before["value"],
                    "predictor": predictor,
                    "baseline": baseline,
                    "candidate": result,
                    "difference": result - baseline,
                }
            )

    groups = {
        "clean": _mean_group(
            rows, member_ids, lambda row: row["family"] == "clean"
        ),
        "awgn_minus5_db": _mean_group(
            rows,
            member_ids,
            lambda row: row["family"] == "awgn" and row["value"] == -5.0,
        ),
        "awgn_0_to_5_db": _mean_group(
            rows,
            member_ids,
            lambda row: row["family"] == "awgn" and row["value"] in [0.0, 5.0],
        ),
        "multipath": _mean_group(
            rows, member_ids, lambda row: row["family"] == "multipath"
        ),
    }
    minus5_rows = [
        row
        for row in rows
        if row["predictor"] in member_ids
        and row["family"] == "awgn"
        and row["value"] == -5.0
    ]
    worst = {
        "baseline": min(float(row["baseline"]) for row in minus5_rows),
        "candidate": min(float(row["candidate"]) for row in minus5_rows),
    }
    worst["difference"] = worst["candidate"] - worst["baseline"]
    checks = {
        "clean_guardrail": groups["clean"]["difference"] >= -0.03,
        "minus5_mean_improved": groups["awgn_minus5_db"]["difference"] > 0.0,
        "minus5_worst_case_improved": worst["difference"] > 0.0,
        "zero_to5_guardrail": groups["awgn_0_to_5_db"]["difference"] >= -0.03,
        "multipath_guardrail": groups["multipath"]["difference"] >= -0.03,
    }
    return {
        "status": "completed_validation_development_comparison",
        "acceptance_passed": all(checks.values()),
        "acceptance_checks": checks,
        "primary_single_model_summaries": groups,
        "minus5_worst_seed_repeat": worst,
        "paired_rows": rows,
        "test_data_used": False,
        "unknown_data_used": False,
        "open_set_threshold_calibration_used": False,
        "competition_test_claim_allowed": False,
        "limitation": (
            "The same 24-member development validation split and stress repeats "
            "are now used for checkpoint selection and comparison; this is not "
            "independent generalization evidence."
        ),
    }


def main() -> None:
    protocol = load_v3_protocol()
    if COMPARISON_PATH.exists():
        raise FileExistsError(COMPARISON_PATH)

    results = []
    for seed in protocol["seeds"]:
        config = load_config(ROOT / protocol["base_config"])
        config["augmentation"] = protocol["augmentation"]
        config["checkpoint_selection"] = protocol["checkpoint_selection"]
        config["training"]["seed"] = seed
        run_name = f"ku_leuven_tcn_worst_noise_selection_seed{seed}_v3"
        config["output"]["run_name"] = run_name
        evidence_path = EVIDENCE / f"{run_name}_training.json"
        if evidence_path.exists():
            raise FileExistsError(evidence_path)
        result = train_ku_leuven(config, run_name=run_name)
        results.append((result, evidence_path))

    template_path = (
        ROOT / "training/configs/ku_leuven_tcn_local_validation_robustness_v1.json"
    )
    evaluation = json.loads(template_path.read_text(encoding="utf-8"))
    for specification, (result, evidence_path) in zip(
        evaluation["models"], results
    ):
        for field, path in (
            ("training_evidence", evidence_path),
            ("resolved_config", Path(result["resolved_config_path"])),
            ("checkpoint", Path(result["checkpoint_path"])),
        ):
            specification[field] = path.relative_to(ROOT).as_posix()
            specification[field + "_sha256"] = sha256_file(path)
    evaluation["purpose"] = (
        "Controlled clean-guarded worst--5-dB checkpoint-selection candidate "
        "evaluated on the unchanged development-only 40-condition protocol."
    )
    for key, value in evaluation["outputs"].items():
        evaluation["outputs"][key] = value.replace(
            "tcn_validation_robustness_v1",
            "tcn_worst_noise_selection_validation_robustness_v3",
        )
    evaluation_path = ROOT / (
        "training/configs/"
        "ku_leuven_tcn_worst_noise_selection_validation_robustness_v3.json"
    )
    if evaluation_path.exists():
        raise FileExistsError(evaluation_path)
    write_json(evaluation_path, evaluation)
    candidate = evaluate(evaluation_path)

    prior = json.loads(PRIOR_EVIDENCE_PATH.read_text(encoding="utf-8"))
    comparison = compare(prior, candidate)
    comparison.update(
        {
            "protocol": protocol["protocol"],
            "controlled_change": protocol["controlled_change"],
            "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prior_evidence_path": PRIOR_EVIDENCE_PATH.relative_to(ROOT).as_posix(),
            "prior_evidence_sha256": sha256_file(PRIOR_EVIDENCE_PATH),
            "candidate_evidence_path": evaluation["outputs"]["summary_json"],
            "candidate_evidence_sha256": sha256_file(
                ROOT / evaluation["outputs"]["summary_json"]
            ),
            "checkpoint_selection": [
                result["best_checkpoint_selection"] for result, _ in results
            ],
        }
    )
    write_json(COMPARISON_PATH, comparison)
    print(
        json.dumps(
            {
                key: value
                for key, value in comparison.items()
                if key != "paired_rows"
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
