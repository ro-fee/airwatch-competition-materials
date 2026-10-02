"""Run the controlled -5 dB training experiment and paired validation audit."""

import json
from pathlib import Path
from statistics import fmean

from airwatch.data.data_provenance import sha256_file
from training.common import load_config, write_json
from training.evaluate_ku_leuven_robustness import run as evaluate
from training.ku_leuven_baseline import train_ku_leuven


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/evidence/uav/ku_leuven/local_development"
PROTOCOL_PATH = ROOT / "training/configs/ku_leuven_strong_noise_local_protocol_v2.json"
PRIOR_EVIDENCE_PATH = EVIDENCE / "ku_leuven_tcn_augmented_validation_robustness_v1.json"
COMPARISON_PATH = EVIDENCE / "ku_leuven_strong_noise_local_comparison_v2.json"


def _mean_group(rows, member_ids, selector):
    selected = [
        row for row in rows if row["predictor"] in member_ids and selector(row)
    ]
    if not selected:
        raise ValueError("comparison group is empty")
    return {
        key: fmean(row[key] for row in selected)
        for key in ("baseline", "candidate", "difference")
    }


def compare(prior, candidate):
    """Make an exactly paired V1-augmentation versus strong-noise comparison."""
    if (
        [condition["condition_id"] for condition in prior["conditions"]]
        != [condition["condition_id"] for condition in candidate["conditions"]]
        or prior["dataset"] != candidate["dataset"]
    ):
        raise ValueError("paired comparison requires identical conditions and data")
    member_ids = [source["id"] for source in prior["sources"]]
    if member_ids != [source["id"] for source in candidate["sources"]]:
        raise ValueError("paired comparison requires identical predictor identities")
    ensemble_id = prior["ensemble"]["id"]
    if ensemble_id != candidate["ensemble"]["id"]:
        raise ValueError("paired comparison requires identical ensemble identity")

    rows = []
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
        "clean": _mean_group(rows, member_ids, lambda row: row["family"] == "clean"),
        "awgn_minus5_to_5_db": _mean_group(
            rows,
            member_ids,
            lambda row: row["family"] == "awgn" and row["value"] in [-5, 0, 5],
        ),
        "multipath": _mean_group(
            rows, member_ids, lambda row: row["family"] == "multipath"
        ),
    }
    ensemble_groups = {
        "clean": _mean_group(rows, [ensemble_id], lambda row: row["family"] == "clean"),
        "awgn_minus5_to_5_db": _mean_group(
            rows,
            [ensemble_id],
            lambda row: row["family"] == "awgn" and row["value"] in [-5, 0, 5],
        ),
        "multipath": _mean_group(
            rows, [ensemble_id], lambda row: row["family"] == "multipath"
        ),
    }
    checks = {
        "clean_guardrail": groups["clean"]["difference"] >= -0.03,
        "awgn_minus5_to_5_db_improved": groups["awgn_minus5_to_5_db"]["difference"] > 0,
        "multipath_guardrail": groups["multipath"]["difference"] >= -0.03,
    }
    return {
        "status": "completed_validation_development_comparison",
        "acceptance_passed": all(checks.values()),
        "acceptance_checks": checks,
        "primary_single_model_summaries": groups,
        "ensemble_diagnostic_summaries": ensemble_groups,
        "paired_rows": rows,
        "test_data_used": False,
        "unknown_data_used": False,
        "competition_test_claim_allowed": False,
        "limitation": "The same 24-member development validation split has been repeatedly used; this comparison cannot support an independent generalization claim.",
    }


def _verify_references(protocol):
    for name in ("prior_protocol", "prior_robustness", "prior_comparison"):
        relative = protocol["reference_evidence"][name]
        expected = protocol["reference_evidence"][name + "_sha256"]
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"reference hash mismatch for {relative}: {actual}")


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_references(protocol)
    if COMPARISON_PATH.exists():
        raise FileExistsError(COMPARISON_PATH)

    results = []
    for seed in protocol["seeds"]:
        config = load_config(ROOT / protocol["base_config"])
        config["augmentation"] = protocol["augmentation"]
        config["training"]["seed"] = seed
        run_name = f"ku_leuven_tcn_strong_noise_seed{seed}_v2"
        config["output"]["run_name"] = run_name
        evidence_path = EVIDENCE / f"{run_name}_training.json"
        if evidence_path.exists():
            raise FileExistsError(evidence_path)
        result = train_ku_leuven(config, run_name=run_name)
        results.append((result, evidence_path))

    original_path = ROOT / "training/configs/ku_leuven_tcn_local_validation_robustness_v1.json"
    evaluation = json.loads(original_path.read_text(encoding="utf-8"))
    for specification, (result, evidence_path) in zip(evaluation["models"], results):
        for field, path in (
            ("training_evidence", evidence_path),
            ("resolved_config", Path(result["resolved_config_path"])),
            ("checkpoint", Path(result["checkpoint_path"])),
        ):
            specification[field] = path.relative_to(ROOT).as_posix()
            specification[field + "_sha256"] = sha256_file(path)
    evaluation["purpose"] = (
        "Controlled strong-noise TCN candidate evaluated on the unchanged "
        "development-only 40-condition protocol."
    )
    for key, value in evaluation["outputs"].items():
        evaluation["outputs"][key] = value.replace(
            "tcn_validation_robustness_v1",
            "tcn_strong_noise_validation_robustness_v2",
        )
    evaluation_path = (
        ROOT / "training/configs/ku_leuven_tcn_strong_noise_validation_robustness_v2.json"
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
            "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prior_evidence_path": PRIOR_EVIDENCE_PATH.relative_to(ROOT).as_posix(),
            "prior_evidence_sha256": sha256_file(PRIOR_EVIDENCE_PATH),
            "candidate_evidence_path": evaluation["outputs"]["summary_json"],
            "candidate_evidence_sha256": sha256_file(
                ROOT / evaluation["outputs"]["summary_json"]
            ),
        }
    )
    write_json(COMPARISON_PATH, comparison)
    print(
        json.dumps(
            {key: value for key, value in comparison.items() if key != "paired_rows"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
