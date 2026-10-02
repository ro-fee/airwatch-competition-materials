"""Post-hoc error diagnosis from immutable validation prediction artifacts."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from airwatch.data.data_provenance import sha256_file
from training.common import write_json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/evidence/uav/ku_leuven/local_development'
ENSEMBLE = 'tcn_three_seed_probability_ensemble'
LABELS = ['frysky', 'spektrum_dx4e', 'dji_mini2_rc']


def summarize(rows):
    if not rows:
        raise ValueError('empty prediction group')
    errors = [r for r in rows if r['target'] != r['prediction']]
    matrix = [[0] * 3 for _ in range(3)]
    for row in rows:
        matrix[row['target']][row['prediction']] += 1
    return {
        'count': len(rows), 'error_count': len(errors),
        'error_fraction': len(errors) / len(rows),
        'mean_confidence': fmean(r['confidence'] for r in rows),
        'mean_error_confidence': fmean(r['confidence'] for r in errors) if errors else None,
        'errors_with_confidence_at_least_0_9': sum(r['confidence'] >= 0.9 for r in errors),
        'prediction_counts': dict(Counter(LABELS[r['prediction']] for r in rows)),
        'confusion_matrix': matrix,
        'per_class_recall': {
            name: matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None
            for i, name in enumerate(LABELS)
        },
    }


def load_predictions(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        for key in ('target', 'prediction'):
            row[key] = int(row[key])
            if row[key] not in range(3):
                raise ValueError('invalid class')
        row['confidence'] = float(row['confidence'])
        probabilities = [float(row['prob_' + name]) for name in LABELS]
        if (not 0 <= row['confidence'] <= 1
                or any(not 0 <= p <= 1 for p in probabilities)
                or abs(sum(probabilities) - 1) > 1e-5
                or abs(max(probabilities) - row['confidence']) > 1e-6):
            raise ValueError('invalid probability vector')
        key = (row['condition_id'], row['predictor_id'], row['recording_id'])
        if key in seen:
            raise ValueError('duplicate prediction')
        seen.add(key)
    return rows


def build():
    output = DIRECTORY / 'ku_leuven_noise_error_diagnosis_v1.json'
    if output.exists():
        raise FileExistsError(output)
    report = {'artifact_type': 'ku_leuven_post_hoc_noise_error_diagnosis',
              'label_order': LABELS, 'sources': {}, 'results': {},
              'test_data_used': False, 'unknown_data_used': False,
              'competition_test_claim_allowed': False,
              'diagnostic_confidence_cutoff': 0.9,
              'cutoff_is_not_an_open_set_threshold': True,
              'limitation': 'Repeats reuse 24 validation members; pooled counts are not independent recordings. Descriptive diagnosis, not causal attribution.'}
    for name, stem in (
        ('baseline', 'ku_leuven_tcn_validation_robustness_v1'),
        ('augmented', 'ku_leuven_tcn_augmented_validation_robustness_v1'),
    ):
        evidence_path = DIRECTORY / (stem + '.json')
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        if evidence.get('test_data_used') is not False or evidence.get('unknown_data_used') is not False:
            raise ValueError('not sealed validation evidence')
        spec = evidence['artifacts']['recording_predictions_csv']
        path = Path(spec['path'])
        if sha256_file(path) != spec['sha256']:
            raise ValueError('prediction hash mismatch')
        rows = load_predictions(path)
        report['sources'][name] = {'evidence': str(evidence_path),
                                  'evidence_sha256': sha256_file(evidence_path),
                                  'prediction_sha256': spec['sha256']}
        groups = defaultdict(list)
        for row in rows:
            if row['family'] in ('clean', 'awgn'):
                groups[(row['family'], row['value'], row['predictor_id'])].append(row)
        statistics = [{'family': family, 'value': value, 'predictor_id': predictor,
                       **summarize(group)}
                      for (family, value, predictor), group in groups.items()]
        by_case = defaultdict(dict)
        for row in rows:
            if row['family'] == 'awgn':
                by_case[(row['condition_id'], row['recording_id'])][row['predictor_id']] = row
        ensemble_cases = defaultdict(list)
        for key, predictions in by_case.items():
            if len(predictions) != 4 or ENSEMBLE not in predictions:
                raise ValueError('incomplete ensemble case')
            ensemble = predictions[ENSEMBLE]
            members = [r for p, r in predictions.items() if p != ENSEMBLE]
            if len({r['target'] for r in predictions.values()}) != 1:
                raise ValueError('inconsistent paired targets')
            ensemble_cases[ensemble['value']].append({
                'ensemble_wrong': ensemble['target'] != ensemble['prediction'],
                'all_members_wrong': all(r['target'] != r['prediction'] for r in members),
                'any_member_correct': any(r['target'] == r['prediction'] for r in members),
                'members_unanimous': len({r['prediction'] for r in members}) == 1,
            })
        agreement = {value: {
            'count': len(cases),
            'all_members_wrong': sum(c['all_members_wrong'] for c in cases),
            'ensemble_wrong_despite_any_member_correct': sum(
                c['ensemble_wrong'] and c['any_member_correct'] for c in cases),
            'members_unanimous': sum(c['members_unanimous'] for c in cases),
        } for value, cases in ensemble_cases.items()}
        report['results'][name] = {'group_statistics': statistics, 'ensemble_agreement': agreement}
    write_json(output, report)
    for name, result in report['results'].items():
        selected = [r for r in result['group_statistics']
                    if r['predictor_id'] == ENSEMBLE and r['value'] in ('-5.0', '0.0', '5.0')]
        print(json.dumps({'model': name, 'ensemble': selected,
                          'agreement': result['ensemble_agreement'].get('-5.0')}, indent=2))


if __name__ == '__main__':
    build()
