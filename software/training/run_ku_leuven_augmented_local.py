"""Execute the frozen local augmentation experiment and paired validation audit."""
import json
from pathlib import Path
from statistics import fmean

from airwatch.data.data_provenance import sha256_file
from training.common import load_config, write_json
from training.ku_leuven_baseline import train_ku_leuven
from training.evaluate_ku_leuven_robustness import run as evaluate

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/evidence/uav/ku_leuven/local_development"


def compare(baseline, candidate):
    if [c['condition_id'] for c in baseline['conditions']] != [
        c['condition_id'] for c in candidate['conditions']
    ] or baseline['dataset'] != candidate['dataset']:
        raise ValueError('paired comparison requires identical conditions and data')
    member_ids = [s['id'] for s in baseline['sources']]
    rows = []
    for before, after in zip(baseline['conditions'], candidate['conditions']):
        for member in member_ids + [baseline['ensemble']['id']]:
            b = before['metrics'][member]['recording']['macro_f1']
            a = after['metrics'][member]['recording']['macro_f1']
            rows.append(dict(condition_id=before['condition_id'], family=before['family'],
                             value=before['value'], predictor=member,
                             baseline=b, candidate=a, difference=a-b))
    groups = {}
    for name, select in (
        ('clean', lambda r: r['family'] == 'clean'),
        ('awgn_0_10_db', lambda r: r['family'] == 'awgn' and r['value'] in [0, 5, 10]),
        ('multipath', lambda r: r['family'] == 'multipath'),
    ):
        selected = [r for r in rows if r['predictor'] in member_ids and select(r)]
        groups[name] = {k: fmean(r[k] for r in selected)
                        for k in ('baseline', 'candidate', 'difference')}
    accepted = (groups['clean']['difference'] >= -0.03
                and groups['awgn_0_10_db']['difference'] > 0
                and groups['multipath']['difference'] > 0)
    return dict(status='completed_validation_development_comparison',
                acceptance_passed=accepted, summaries=groups, paired_rows=rows,
                test_data_used=False, unknown_data_used=False,
                competition_test_claim_allowed=False,
                limitation='Repeatedly used development validation; no independent generalization claim.')


def main():
    protocol_path = ROOT / 'training/configs/ku_leuven_augmented_local_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    destination = EVIDENCE / 'ku_leuven_augmented_local_comparison_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    results = []
    for seed in protocol['seeds']:
        config = load_config(ROOT / protocol['base_config'])
        config['augmentation'] = protocol['augmentation']
        config['training']['seed'] = seed
        name = f'ku_leuven_tcn_augmented_seed{seed}_v1'
        config['output']['run_name'] = name
        evidence_path = EVIDENCE / f'{name}_training.json'
        # A partial experiment is intentionally not silently reused.
        if evidence_path.exists():
            raise FileExistsError(evidence_path)
        result = train_ku_leuven(config, run_name=name)
        results.append((result, evidence_path))
    original_protocol = ROOT / 'training/configs/ku_leuven_tcn_local_validation_robustness_v1.json'
    evaluation = json.loads(original_protocol.read_text(encoding='utf-8'))
    for spec, (result, evidence_path) in zip(evaluation['models'], results):
        for field, path in (
            ('training_evidence', evidence_path),
            ('resolved_config', Path(result['resolved_config_path'])),
            ('checkpoint', Path(result['checkpoint_path'])),
        ):
            spec[field] = path.relative_to(ROOT).as_posix()
            spec[field + '_sha256'] = sha256_file(path)
    evaluation['purpose'] = 'Frozen augmented TCN candidate evaluated on the unchanged development conditions.'
    for key, value in evaluation['outputs'].items():
        evaluation['outputs'][key] = value.replace('tcn_validation_robustness_v1', 'tcn_augmented_validation_robustness_v1')
    eval_path = ROOT / 'training/configs/ku_leuven_tcn_augmented_validation_robustness_v1.json'
    if eval_path.exists():
        raise FileExistsError(eval_path)
    write_json(eval_path, evaluation)
    candidate = evaluate(eval_path)
    baseline_path = EVIDENCE / 'ku_leuven_tcn_validation_robustness_v1.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    comparison = compare(baseline, candidate)
    comparison['protocol_sha256'] = sha256_file(protocol_path)
    comparison['baseline_evidence_sha256'] = sha256_file(baseline_path)
    comparison['candidate_evidence_sha256'] = sha256_file(ROOT / evaluation['outputs']['summary_json'])
    write_json(destination, comparison)
    print(json.dumps({k:v for k,v in comparison.items() if k != 'paired_rows'}, indent=2))


if __name__ == '__main__':
    main()
