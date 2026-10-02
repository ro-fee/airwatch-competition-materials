"""Run V4 with deterministic -5 dB weighting and the frozen V3 selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from airwatch.data.data_provenance import sha256_file
from training.common import TrainingConfigError, load_config, write_json
from training.evaluate_ku_leuven_robustness import run as evaluate
from training.ku_leuven_baseline import train_ku_leuven
from training.ku_leuven_checkpoint_selection import validate_checkpoint_selection
from training.run_ku_leuven_worst_noise_selection_local import compare


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/evidence/uav/ku_leuven/local_development"
PROTOCOL_PATH = (
    ROOT / "training/configs/ku_leuven_weighted_worst_noise_local_protocol_v4.json"
)
PRIOR_EVIDENCE_PATH = (
    EVIDENCE / "ku_leuven_tcn_worst_noise_selection_validation_robustness_v3.json"
)
COMPARISON_PATH = EVIDENCE / "ku_leuven_weighted_worst_noise_local_comparison_v4.json"
EXPECTED_PROTOCOL = "ku-leuven-weighted-worst-noise-local-development-v4"
EXPECTED_SEEDS = [20260909, 20260910, 20260911]
EXPECTED_AUGMENTATION = (
    "ku-leuven-clean-weighted-minus5-awgn-multipath-training-v3"
)


def load_v4_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取 V4 协议：{source}: {exc}") from exc
    if not isinstance(protocol, dict) or protocol.get("protocol") != EXPECTED_PROTOCOL:
        raise TrainingConfigError("V4 协议标识不匹配")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise TrainingConfigError("V4 必须使用冻结的三个随机种子")
    if protocol.get("augmentation") != {"protocol": EXPECTED_AUGMENTATION}:
        raise TrainingConfigError("V4 加权噪声增强协议不匹配")
    validate_checkpoint_selection(protocol.get("checkpoint_selection"))
    if protocol.get("development_validation_only") is not True or any(
        protocol.get(name) is not False
        for name in (
            "test_data_allowed",
            "unknown_data_allowed",
            "open_set_threshold_calibration_allowed",
            "competition_test_claim_allowed",
            "a800_protocol_replaced",
        )
    ):
        raise TrainingConfigError("V4 数据封存或声明边界不满足要求")
    references = protocol.get("reference_evidence")
    if not isinstance(references, dict):
        raise TrainingConfigError("V4 缺少参考证据")
    for name in ("base_config", "v3_protocol", "v3_robustness", "v3_comparison"):
        relative = references.get(name)
        expected = references.get(name + "_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TrainingConfigError(f"V4 参考证据 {name} 不完整")
        target = (ROOT / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise TrainingConfigError(f"V4 参考证据哈希不匹配：{relative}")
    protocol["_protocol_path"] = str(source)
    return protocol


def main() -> None:
    protocol = load_v4_protocol()
    if COMPARISON_PATH.exists():
        raise FileExistsError(COMPARISON_PATH)

    results = []
    for seed in protocol["seeds"]:
        config = load_config(ROOT / protocol["base_config"])
        config["augmentation"] = protocol["augmentation"]
        config["checkpoint_selection"] = protocol["checkpoint_selection"]
        config["training"]["seed"] = seed
        run_name = f"ku_leuven_tcn_weighted_worst_noise_seed{seed}_v4"
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
        "Controlled weighted--5-dB training candidate evaluated on the unchanged "
        "development-only 40-condition protocol."
    )
    for key, value in evaluation["outputs"].items():
        evaluation["outputs"][key] = value.replace(
            "tcn_validation_robustness_v1",
            "tcn_weighted_worst_noise_validation_robustness_v4",
        )
    evaluation_path = ROOT / (
        "training/configs/"
        "ku_leuven_tcn_weighted_worst_noise_validation_robustness_v4.json"
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
            "training_mixture": protocol["training_mixture"],
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
