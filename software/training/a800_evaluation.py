"""A800 V2 member-level validation, immutable freeze, and sealed evaluation.

Numerical helpers accept CPU fixtures. Formal entry points always require a live
allocation context. Files and process checks provide auditable workflow isolation,
not a security boundary against another process running as the same user.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from airwatch.analysis.iq_perturbations import (
    DeterministicIQPerturbationDataset, IQPerturbationSpec,
)
from airwatch.data.data_provenance import sha256_file
from airwatch.inference.uav_open_set import aggregate_recording_probabilities, energy_knownness

SEEDS = [20260909, 20260910, 20260911]
REPEATS = [2026090901, 2026090902, 2026090903]
V1_ARTIFACT = 'ku_leuven_materialized_known_train_validation'
SCORES = ('cosine', 'msp', 'energy')
LABEL_MAP = {'frysky': 0, 'spektrum_dx4e': 1, 'dji_mini2_rc': 2}
_VERIFIED_ALLOCATIONS = {}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def atomic_json(path, payload):
    from training.a800_common import atomic_json as durable_json
    durable_json(Path(path), payload)


def relative_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes experiment root: {value}')
    return path


def check_allocation(allocation, device=None):
    if device is not None and torch.device(device).type == 'cpu':
        return
    if allocation is None:
        raise ValueError('A live explicit GPU allocation is required')
    if allocation.should_stop():
        raise InterruptedError('Allocation ending; preserve completed shards and resume next allocation')


def start_finalization(allocation):
    from training.a800_common import Allocation
    if not isinstance(allocation, Allocation) or allocation.phase != 'finalize':
        raise ValueError('Formal finalization requires an allocated finalize phase')
    allocation.recheck()
    if id(allocation) not in _VERIFIED_ALLOCATIONS:
        from training.build_a800_bundle import verify_integrity
        from training.a800_environment import verify_environment
        verify_integrity(allocation.root)
        verify_environment(allocation.root, require_linux=True)
        _VERIFIED_ALLOCATIONS[id(allocation)] = allocation
    check_allocation(allocation)
    torch.cuda.set_device(allocation.gpu_index)


def _matrix(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite nonempty matrix')
    return array


def _unit(values):
    values = _matrix(values, 'features')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError('Zero/nonfinite feature norm')
    return values / norms


def aggregate_members(logits, features, labels, recording_ids, prototypes=None):
    logits = _matrix(logits, 'logits')
    features = _unit(features)
    if features.shape[0] != logits.shape[0] or logits.shape[1] != 3:
        raise ValueError('Window feature/logit shape mismatch')
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    identifiers, mean_probabilities, targets = aggregate_recording_probabilities(
        probabilities, recording_ids, labels, expected_windows_per_recording=32)
    ids = np.asarray(recording_ids)
    energies = energy_knownness(logits, temperature=1.)
    vectors = _unit(np.vstack([features[ids == rid].mean(axis=0) for rid in identifiers]))
    knownness = None if prototypes is None else vectors @ _unit(prototypes).T
    result = []
    for i, identifier in enumerate(identifiers):
        row = dict(recording_id=identifier, label=int(targets[i]), prediction=int(mean_probabilities[i].argmax()),
                   probabilities=mean_probabilities[i].tolist(), msp=float(mean_probabilities[i].max()),
                   energy=float(energies[ids == identifier].mean()), feature=vectors[i].tolist(), windows=32)
        if knownness is not None:
            row['cosine'] = float(knownness[i].max())
        result.append(row)
    return result


def fit_prototypes(features, labels, recording_ids, *, split, artifact_type):
    if split != 'train' or artifact_type != V1_ARTIFACT:
        raise ValueError('Prototypes require original clean V1 train windows exclusively')
    features = _matrix(features, 'features')
    rows = aggregate_members(np.zeros((len(features), 3)), features, labels, recording_ids)
    targets = np.asarray([r['label'] for r in rows])
    if set(targets) != {0, 1, 2}:
        raise ValueError('Prototype fitting requires all three registered known classes')
    member_features = np.asarray([r['feature'] for r in rows])
    return _unit(np.vstack([member_features[targets == label].mean(axis=0) for label in range(3)]))


def calibrate_scores(scores, *, split):
    values = np.asarray(scores, dtype=np.float64)
    if split != 'validation' or values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError('Calibration requires exactly 24 finite known validation member scores')
    threshold = float(np.sort(values)[1])
    accepted = int((values >= threshold).sum())
    return dict(threshold=threshold, accepted=accepted, members=24, actual_acceptance=accepted / 24,
                target_acceptance=.95, required_acceptance_count=23, comparison='score>=threshold',
                direction='higher_is_known', source_split='validation')


def open_set_metrics(known, unknown, threshold):
    known, unknown = np.asarray(known, float), np.asarray(unknown, float)
    if known.ndim != 1 or unknown.ndim != 1 or min(known.size, unknown.size) < 1:
        raise ValueError('Known and unknown scores must be nonempty vectors')
    if not np.isfinite(np.r_[known, unknown, threshold]).all():
        raise ValueError('Nonfinite open-set scores')
    labels = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    fpr, tpr, _ = roc_curve(labels, np.r_[known, unknown], drop_intermediate=False)
    return dict(auroc=float(roc_auc_score(labels, np.r_[known, unknown])),
                fpr95=float(fpr[tpr >= .95].min()), known_acceptance=float((known >= threshold).mean()),
                unknown_false_acceptance=float((unknown >= threshold).mean()),
                unknown_rejection=float((unknown < threshold).mean()), threshold=float(threshold),
                known_positive=True, roc={'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
                fpr95_definition='minimum observed FPR at TPR>=0.95; no interpolation; descriptive only')


def checkpoint_rank(metrics):
    clean = float(metrics['clean_recording_macro_f1'])
    minus5 = list(map(float, metrics['minus5_recording_macro_f1']))
    loss = float(metrics['clean_window_loss'])
    if len(minus5) != 3 or not np.isfinite([clean, loss, *minus5]).all():
        raise ValueError('Incomplete or nonfinite checkpoint validation')
    if any(x < 0 or x > 1 for x in [clean, *minus5]) or loss < 0:
        raise ValueError('Invalid checkpoint metrics')
    return [min(minus5), fmean(minus5), clean, -loss] if clean >= .97 else None


def select_checkpoint(history):
    eligible = [r for r in history if checkpoint_rank(r) is not None]
    if not eligible:
        raise ValueError('failed_no_eligible_checkpoint')
    return max(eligible, key=lambda r: (*checkpoint_rank(r), -int(r['step'])))


def select_experiment(groups):
    baseline = groups.get('A', {})
    if not baseline.get('seeds_complete'):
        raise ValueError('All three eligible A controls are mandatory')
    eligible, rejected = [], {}
    for name in 'BCD':
        candidate = groups.get(name, {})
        if not candidate.get('seeds_complete'):
            rejected[name] = 'incomplete_eligible_seed_group'
            continue
        safeguards = all(candidate[k] >= baseline[k] - .03 for k in ('clean', 'zero_five', 'multipath'))
        if safeguards and candidate['minus5_mean'] > baseline['minus5_mean'] and candidate['minus5_worst'] >= baseline['minus5_worst']:
            eligible.append(name)
        else:
            rejected[name] = 'preregistered_improvement_gate_not_met'
    selected = max(eligible, key=lambda k: (groups[k]['minus5_worst'], groups[k]['minus5_mean'], groups[k]['clean'], -groups[k]['parameters'])) if eligible else 'A'
    return dict(selected_experiment=selected, deployment_seed=20260909, eligible=eligible, rejected=rejected,
                negative_result=not bool(eligible), groups=groups, source_split='validation')


def conditions():
    result = [dict(id='clean', family='clean', value=None, repeat_seed=None, spec=None)]
    for repeat_index, seed in enumerate(REPEATS, 1):
        for db in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
            result.append(dict(id=f'awgn_{db}_r{repeat_index}', family='awgn', value=db, repeat_seed=seed, spec={'kind': 'awgn', 'snr_db': db}))
    for offset in [-100000, -50000, -25000, 25000, 50000, 100000]:
        result.append(dict(id=f'cfo_{offset}', family='cfo', value=offset, repeat_seed=0, spec={'kind': 'carrier_frequency_offset', 'offset_hz': offset, 'sample_rate_hz': 100000000}))
    for repeat_index, seed in enumerate(REPEATS, 1):
        for profile in ('mild', 'severe'):
            result.append(dict(id=f'multipath_{profile}_r{repeat_index}', family='multipath', value=profile, repeat_seed=seed, spec={'kind': 'multipath', 'profile': profile}))
    return result


def classification(rows):
    targets = np.asarray([x['label'] for x in rows])
    predictions = np.asarray([x['prediction'] for x in rows])
    if len(rows) == 0 or not set(targets).issubset({0, 1, 2}):
        raise ValueError('Known classification requires registered labels')
    return dict(macro_f1=float(f1_score(targets, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
                accuracy=float(accuracy_score(targets, predictions)), members=len(rows),
                confusion_matrix=confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist())


def paired_bootstrap(labels, baseline_predictions, candidate_predictions, *, replicates=2000, seed=2026090904, allocation=None):
    labels, base, candidate = map(np.asarray, (labels, baseline_predictions, candidate_predictions))
    if labels.ndim != 1 or labels.shape != base.shape or labels.shape != candidate.shape or not len(labels):
        raise ValueError('Paired bootstrap needs aligned member predictions')
    if replicates < 1:
        raise ValueError('Bootstrap replicates must be positive')
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    generator, deltas = np.random.default_rng(seed), []
    for iteration in range(replicates):
        if allocation is not None and iteration % 50 == 0:
            check_allocation(allocation)
        ix = np.concatenate([generator.choice(group, len(group), replace=True) for group in groups])
        a = f1_score(labels[ix], base[ix], labels=[0, 1, 2], average='macro', zero_division=0)
        b = f1_score(labels[ix], candidate[ix], labels=[0, 1, 2], average='macro', zero_division=0)
        deltas.append(float(b - a))
    return dict(delta_macro_f1_ci95=np.quantile(deltas, [.025, .975]).tolist(), replicates=replicates, seed=seed,
                unit='class-stratified paired MAT member', limitation='Physical acquisition-session independence is not established')


def classification_bootstrap(rows):
    """Class-stratified member resampling; identical member keys pair all runs."""
    rows = sorted(rows, key=lambda r: r['recording_id'])
    labels = np.asarray([r['label'] for r in rows])
    predictions = np.asarray([r['prediction'] for r in rows])
    if set(labels) != {0, 1, 2}:
        raise ValueError('Three known classes required for stratified intervals')
    generator = np.random.default_rng(2026090904)
    groups = [np.flatnonzero(labels == i) for i in range(3)]
    draws = np.concatenate([generator.choice(g, (2000, len(g)), replace=True) for g in groups], axis=1)
    actual, predicted = labels[draws], predictions[draws]
    f1s = []
    for label in range(3):
        tp = ((actual == label) & (predicted == label)).sum(axis=1)
        denominator = (actual == label).sum(axis=1) + (predicted == label).sum(axis=1)
        f1s.append(2 * tp / denominator)
    return dict(macro_f1_ci95=np.quantile(np.mean(f1s, axis=0), [.025, .975]).tolist(),
                accuracy_ci95=np.quantile((actual == predicted).mean(axis=1), [.025, .975]).tolist(),
                replicates=2000, seed=2026090904, unit='class-stratified MAT member',
                limitation='Current member sampling only; physical session independence is not established')


def predict_dataset(model, dataset, device, *, batch_size=64, prototypes=None, allocation=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = torch.device(device)
    check_allocation(allocation, device)
    model.eval().float()
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    logits, features, targets, ids, windows = [], [], [], [], []
    loss_sum = 0.
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            check_allocation(allocation, device)
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            inputs = torch.stack([x[0] for x in items]).to(device=device, dtype=torch.float32)
            feature = model.forward_features(inputs)
            # The frozen models expose a linear classifier; this avoids a second backbone pass.
            output = model.classifier(feature) if hasattr(model, 'classifier') else model(inputs)
            if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(feature).all()):
                raise ValueError('Model returned NaN/Inf')
            batch_targets = np.asarray([int(x[1]) for x in items], dtype=np.int64)
            if (batch_targets >= 0).all():
                loss_sum += float(torch.nn.functional.cross_entropy(output, torch.as_tensor(batch_targets, device=device), reduction='sum').cpu())
            logits.append(output.float().cpu().numpy())
            features.append(feature.float().cpu().numpy())
            targets.extend(batch_targets.tolist())
            for offset, item in enumerate(items):
                meta = dataset.sample_metadata(start + offset)
                get = meta.get if isinstance(meta, dict) else lambda key: getattr(meta, key)
                ids.append(get('recording_id'))
                windows.append(dict(window_id=get('window_id'), recording_id=get('recording_id'), label=int(item[1])))
    logits, features = np.concatenate(logits), np.concatenate(features)
    rows = aggregate_members(logits, features, targets, ids, prototypes)
    for row, output in zip(windows, logits):
        row.update(logits=output.tolist(), prediction=int(output.argmax()))
    return dict(members=rows, windows=windows, logits=logits, features=features, labels=np.asarray(targets),
                recording_ids=ids, window_loss=loss_sum / len(dataset))


def evaluate_validation(model, dataset, device, batch_size=64, *, allocation=None):
    if getattr(dataset, 'split', None) != 'validation':
        raise ValueError('Checkpoint selection permits known validation only')
    chosen = [c for c in conditions() if c['id'] == 'clean' or c['family'] == 'awgn' and c['value'] == -5]
    metrics, detail = [], []
    clean_loss = None
    for condition in chosen:
        view = dataset if condition['spec'] is None else DeterministicIQPerturbationDataset(dataset, perturbation=condition['spec'], base_seed=condition['repeat_seed'])
        output = predict_dataset(model, view, device, batch_size=batch_size, allocation=allocation)
        metrics.append(classification(output['members'])['macro_f1'])
        detail.append(dict(condition=condition, members=output['members'], windows=output['windows']))
        if condition['id'] == 'clean':
            clean_loss = output['window_loss']
    result = dict(clean_recording_macro_f1=metrics[0], minus5_recording_macro_f1=metrics[1:], clean_window_loss=clean_loss, conditions=detail)
    result['rank'] = checkpoint_rank(result)
    return result


def verify_freeze(freeze, *, root=None):
    freeze = Path(freeze).resolve()
    payload = read_json(freeze)
    if payload.get('artifact_type') != 'a800_v2_frozen_experiment' or payload.get('inference_precision') != 'fp32':
        raise ValueError('Missing/invalid immutable A800 freeze')
    experiment_root = Path(root).resolve() if root is not None else relative_path(freeze.parent, payload['root_relative']) if payload['root_relative'] == '.' else (freeze.parent / payload['root_relative']).resolve()
    claimed = payload.get('content_sha256')
    core = {k: v for k, v in payload.items() if k != 'content_sha256'}
    if hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() != claimed:
        raise ValueError('Freeze content digest changed')
    if not payload.get('bindings') or not payload.get('runs') or payload.get('primary_score') != 'cosine':
        raise ValueError('Freeze lacks required experiment bindings')
    if payload.get('schema_version') == '2.0':
        environment_files = sorted(p.relative_to(experiment_root).as_posix() for p in _environment_paths(experiment_root))
        if (payload.get('environment_files') != environment_files
                or not set(environment_files).issubset({item['path'] for item in payload['bindings']})):
            raise ValueError('Freeze lacks immutable environment inputs or installed server snapshots')
    for item in payload['bindings']:
        path = relative_path(experiment_root, item['path'])
        if not path.is_file() or path.stat().st_size != item['size_bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError(f'Frozen artifact changed: {item["path"]}')
    payload['_root'] = str(experiment_root)
    payload['_path'] = str(freeze)
    return payload


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def cached_shard(path, identity, compute):
    """Only atomic, hash-verified complete shards can be reused after interruption."""
    path = Path(path)
    if path.exists():
        stored = read_json(path)
        if stored.get('identity') != identity or stored.get('result_sha256') != _digest(stored.get('result')):
            raise ValueError(f'Evaluation shard identity/content changed: {path}')
        return stored['result']
    try:
        result = compute()
        atomic_json(path, dict(identity=identity, result_sha256=_digest(result), result=result))
        return result
    except Exception as exc:
        atomic_json(path.with_suffix('.failure.json'), dict(identity=identity, error_type=type(exc).__name__, error=str(exc)))
        raise


def _load_model(root, config, checkpoint, device, *, allocation=None):
    from airwatch.models.uav_multiscale_tcn import build_a800_model
    from training.a800_common import canonical_hash, source_identity
    checkpoint = Path(checkpoint)
    from training.a800_runner import load_checkpoint, _datasets, _dataset_identity
    if config.get('kind') == 'historical_reference':
        registered = next((r for r in _historical_runs(root) if r['config'] == config and root / r['checkpoint'] == checkpoint), None)
        if registered is None:
            raise ValueError('Historical reference is not registered by the frozen V4 protocol')
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
        with KULeuvenMaterializedDataset(root / config['data']['validation_root'], split='validation') as dataset:
            if (value.get('model_name') != 'DroneRFTCN' or value.get('label_map') != LABEL_MAP
                    or value.get('data_identity') != dataset.data_identity
                    or value.get('aggregation_protocol') != 'mean-window-softmax-probability-v1'
                    or value.get('config', {}).get('training', {}).get('seed') != config['seed']):
                raise ValueError('Frozen historical V4 checkpoint identity does not match')
        model = build_a800_model('tcn', num_classes=3)
        model.load_state_dict(value['model_state_dict'], strict=True)
        check_allocation(allocation, device)
        return model.float().eval().to(device), value
    value = load_checkpoint(checkpoint)
    if value.get('artifact_type') != 'a800_training_checkpoint' or value.get('kind') != 'formal':
        raise ValueError('Formal validation requires registered A800 V2 checkpoint')
    if value.get('config') != config or value.get('identity', {}).get('config_sha256') != canonical_hash(config):
        raise ValueError('Checkpoint/config identity changed')
    if value['identity']['code'] != source_identity(root):
        raise ValueError('Checkpoint source identity changed')
    train, validation = _datasets(root, config)
    try:
        if value['identity']['data'] != {'train': _dataset_identity(train), 'validation': _dataset_identity(validation)}:
            raise ValueError('Checkpoint train/validation data identity changed')
    finally:
        train.close()
        validation.close()
    model = build_a800_model(config['model']['name'], num_classes=config['model']['num_classes'])
    model.load_state_dict(value['model_state'], strict=True)
    check_allocation(allocation, device)
    return model.float().eval().to(device), value


def _formal_root(config, root):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    if config.get('kind') == 'historical_reference':
        if not any(r['config'] == config for r in _historical_runs(root)):
            raise ValueError('Unregistered historical reference')
    elif config.get('kind') != 'formal' or config.get('protocol') != 'airwatch-a800-v2':
        raise ValueError('Formal finalization rejects smoke/unregistered configurations')
    else:
        from training.a800_runner import validate_config
        validate_config(config)
    return root


def _evaluation_identity(root, config, checkpoint):
    """Bind resumable validation to implementation and all common V1 inputs."""
    from training.a800_common import source_identity
    base = relative_path(root, config['data']['validation_root'])
    paths = [base / 'dataset-metadata.json']
    paths += [base / split / name for split in ('train', 'validation')
              for name in ('data.npy', 'labels.npy', 'windows.csv')]
    return _digest(dict(checkpoint=sha256_file(checkpoint), config=config, code=source_identity(root),
                        data={p.relative_to(root).as_posix(): sha256_file(p) for p in paths}))


def _verify_validation_calibration(root, config, checkpoint, validation, wrapper):
    identity = _evaluation_identity(root, config, checkpoint)
    calibration = wrapper['result']
    if (validation.get('identity') != identity or wrapper.get('identity') != identity
            or wrapper.get('result_sha256') != _digest(calibration)
            or calibration.get('checkpoint_sha256') != sha256_file(checkpoint)
            or validation.get('checkpoint_sha256') != calibration['checkpoint_sha256']
            or validation.get('inference_precision') != 'fp32' or calibration.get('inference_precision') != 'fp32'
            or [r.get('condition') for r in validation.get('conditions', [])] != conditions()):
        raise ValueError('Validation/calibration identity or registered conditions changed')
    for row in validation['conditions']:
        shard = read_json(root / 'outputs/evidence/validation' / config['run_id'] / (row['condition']['id'] + '.json'))
        if shard.get('identity') != identity or shard.get('result_sha256') != _digest(row) or shard.get('result') != row:
            raise ValueError('Validation report differs from completed inference shard')
        if row['metrics'] != classification(row['members']):
            raise ValueError('Validation metrics do not reproduce member predictions')
    for score in SCORES:
        if calibration['scores'][score] != calibrate_scores([r[score] for r in calibration['validation_members']], split='validation'):
            raise ValueError('Calibration threshold does not reproduce validation scores')


def validate_run(config, checkpoint, *, root=None, allocation=None, device='cuda'):
    """Evaluate the preselected checkpoint on all 40 known validation conditions."""
    root = _formal_root(config, root)
    start_finalization(allocation)
    if root != allocation.root.resolve():
        raise ValueError('Validation and allocation refer to different roots')
    from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
    model, _ = _load_model(root, config, checkpoint, device, allocation=allocation)
    identity = _evaluation_identity(root, config, checkpoint)
    output = root / 'outputs/evidence/validation' / config['run_id']
    with KULeuvenMaterializedDataset(relative_path(root, config['data']['validation_root']), split='validation') as dataset:
        if len(dataset) != 768 or len(dataset.recording_ids) != 24:
            raise ValueError('Formal validation must have 24 members and 768 V1 windows')
        rows = []
        for condition in conditions():
            check_allocation(allocation)
            def compute(condition=condition):
                view = dataset if condition['spec'] is None else DeterministicIQPerturbationDataset(dataset, perturbation=condition['spec'], base_seed=condition['repeat_seed'])
                predicted = predict_dataset(model, view, device, allocation=allocation)
                return dict(condition=condition, metrics=classification(predicted['members']), members=predicted['members'], windows=predicted['windows'], window_loss=predicted['window_loss'])
            rows.append(cached_shard(output / (condition['id'] + '.json'), identity, compute))
    report = dict(run_id=config['run_id'], identity=identity, checkpoint_sha256=sha256_file(checkpoint), source_split='validation', conditions=rows,
                  parameters=sum(p.numel() for p in model.parameters()), inference_precision='fp32')
    atomic_json(output / 'validation.json', report)
    del model
    return report


def calibrate_run(config, checkpoint, *, root=None, allocation=None, device='cuda'):
    root = _formal_root(config, root)
    start_finalization(allocation)
    if root != allocation.root.resolve():
        raise ValueError('Calibration and allocation refer to different roots')
    from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
    checkpoint_hash = sha256_file(checkpoint)
    path = root / 'outputs/evidence/calibration' / (config['run_id'] + '.json')
    identity = _evaluation_identity(root, config, checkpoint)
    def compute():
        model, _ = _load_model(root, config, checkpoint, device, allocation=allocation)
        base = relative_path(root, config['data']['validation_root'])
        with KULeuvenMaterializedDataset(base, split='train') as train:
            if len(train) != 3680 or len(train.recording_ids) != 115:
                raise ValueError('Prototype base must be the common original 115 x 32 clean train windows')
            training = predict_dataset(model, train, device, allocation=allocation)
            prototypes = fit_prototypes(training['features'], training['labels'], training['recording_ids'], split=train.split, artifact_type=train.metadata['artifact_type'])
        with KULeuvenMaterializedDataset(base, split='validation') as validation:
            predicted = predict_dataset(model, validation, device, prototypes=prototypes, allocation=allocation)
            thresholds = {score: calibrate_scores([r[score] for r in predicted['members']], split=validation.split) for score in SCORES}
        return dict(run_id=config['run_id'], checkpoint_sha256=checkpoint_hash, prototypes=prototypes.tolist(),
                    prototype_source=dict(split='train', artifact_type=V1_ARTIFACT, members=115, windows_per_member=32, metadata_sha256=sha256_file(base / 'dataset-metadata.json')),
                    scores=thresholds, validation_members=predicted['members'], primary_score='cosine', inference_precision='fp32')
    return cached_shard(path, identity, compute)


def _group_summary(reports):
    groups = {}
    for experiment in 'ABCD':
        runs = [r for r in reports if r['config']['experiment'] == experiment]
        if sorted(r['config']['seed'] for r in runs) != SEEDS:
            groups[experiment] = dict(seeds_complete=False)
            continue
        flattened = [(c['condition'], c['metrics']['macro_f1']) for r in runs for c in r['validation']['conditions']]
        take = lambda pred: [score for condition, score in flattened if pred(condition)]
        minus5 = take(lambda c: c['family'] == 'awgn' and c['value'] == -5)
        groups[experiment] = dict(seeds_complete=True,
            clean=fmean(take(lambda c: c['family'] == 'clean')),
            zero_five=fmean(take(lambda c: c['family'] == 'awgn' and c['value'] in (0, 5))),
            multipath=fmean(take(lambda c: c['family'] == 'multipath')),
            minus5_mean=fmean(minus5), minus5_worst=min(minus5), parameters=runs[0]['validation']['parameters'])
    return groups


def _matrix_root(matrix, root=None):
    matrix = Path(matrix).resolve()
    return Path(root).resolve() if root is not None else matrix.parents[3]


def _historical_runs(root):
    protocol_path = root / 'training/configs/ku_leuven_tcn_weighted_worst_noise_validation_robustness_v4.json'
    protocol = read_json(protocol_path)
    if [row.get('seed') for row in protocol.get('models', [])] != SEEDS:
        raise ValueError('Historical V4 reference must contain all three frozen seeds')
    result = []
    for source in protocol['models']:
        for key in ('checkpoint', 'training_evidence', 'resolved_config'):
            path = relative_path(root, source[key])
            if sha256_file(path) != source[key + '_sha256']:
                raise ValueError(f'Historical V4 {key} identity changed')
        evidence = read_json(root / source['training_evidence'])
        if evidence.get('test_data_used') is not False or evidence.get('unknown_data_used') is not False:
            raise ValueError('Historical evidence does not establish sealed-data boundary')
        config = dict(kind='historical_reference', protocol='frozen-ku-leuven-v4-reference', run_id=f'V4-seed{source["seed"]}',
                      experiment='V4', seed=source['seed'], model={'name': 'tcn', 'num_classes': 3},
                      data={'validation_root': 'data/prepared/known-iq-v1'},
                      budget_comparable_to_a800_v2=False)
        result.append(dict(config=config, checkpoint=source['checkpoint'], reference=source,
                           protocol_path=protocol_path.relative_to(root).as_posix(), budget_disclosure='Historical V4 training budget, views and checkpoint selection differ from A800 V2.'))
    return result


def _completed_runs(root, matrix):
    from training.a800_common import canonical_hash, source_identity
    from training.a800_runner import load_checkpoint, _datasets, _dataset_identity
    registered = read_json(matrix)
    configs = registered.get('runs', [])
    if [(c.get('experiment'), c.get('seed')) for c in configs] != [(e, s) for e in 'ABCD' for s in SEEDS]:
        raise ValueError('All 12 preregistered matrix runs are required')
    eligible, excluded = [], []
    code = source_identity(root)
    data_identities = {}
    for config in configs:
        _formal_root(config, root)
        directory = root / 'outputs/runs' / config['run_id']
        status = read_json(directory / 'run-status.json')
        if status.get('config_sha256') != canonical_hash(config) or status.get('global_step') != 9200:
            raise ValueError(f'Incomplete or changed formal run: {config["run_id"]}')
        final_checkpoint = load_checkpoint(directory / 'last.pt')
        data_key = _digest(config['data'])
        if data_key not in data_identities:
            train, validation = _datasets(root, config)
            try:
                data_identities[data_key] = {'train': _dataset_identity(train), 'validation': _dataset_identity(validation)}
            finally:
                train.close()
                validation.close()
        identity = final_checkpoint['identity']
        if (final_checkpoint['global_step'] != 9200 or final_checkpoint['config'] != config
                or final_checkpoint['kind'] != 'formal' or final_checkpoint['validation_pending'] is not False
                or identity['code'] != code or identity['data'] != data_identities[data_key]
                or identity['precision'] != config['training']['precision']
                or status.get('checkpoint_sha256') != sha256_file(directory / 'last.pt')):
            raise ValueError('Final training checkpoint/code/config/data does not match completed status')
        validations = [row for row in final_checkpoint['history'] if row.get('event') == 'validation']
        if [row['global_step'] for row in validations] != list(range(230, 9201, 230)):
            raise ValueError('Completed run lacks all 40 committed checkpoint validations')
        ranks = [(checkpoint_rank(row['metrics']), row['global_step']) for row in validations]
        ranked = [(rank, step) for rank, step in ranks if rank is not None]
        audit = read_json(directory / 'data-access-audit.json')
        if (audit.get('sealed_access_count') != 0 or audit.get('test_accessed') is not False
                or audit.get('unknown_accessed') is not False or set(audit.get('splits', [])) != {'train', 'validation'}
                or audit.get('inputs') != identity['data'] or audit.get('config_sha256') != identity['config_sha256']):
            raise ValueError('Completed run data-access audit does not match checkpoint identity')
        state = status.get('state', status.get('status'))
        if state == 'failed' and status.get('reason') == 'failed_no_eligible_checkpoint':
            if ranked or any(final_checkpoint[key] is not None for key in ('best_rank', 'best_step', 'best_state')):
                raise ValueError('No-eligible exclusion contradicts committed validation history')
            if config['experiment'] == 'A':
                raise ValueError('Baseline A lacks an eligible seed; stop formal comparison')
            excluded.append(dict(config=config, status=status))
            continue
        checkpoint = directory / 'best.pt'
        if state != 'completed' or not checkpoint.is_file() or status.get('best_checkpoint_sha256') != sha256_file(checkpoint):
            raise ValueError(f'Run is failed, incomplete, or checkpoint changed: {config["run_id"]}')
        best = load_checkpoint(checkpoint, expected_identity=identity)
        if not ranked:
            raise ValueError('Completed run has no eligible checkpoint')
        rank, step = max(ranked, key=lambda item: (*item[0], -item[1]))
        embedded = final_checkpoint['best_state']
        if (final_checkpoint['best_rank'] != rank or final_checkpoint['best_step'] != step
                or best['global_step'] != step or best['config'] != config or best['kind'] != 'formal'
                or embedded is None or set(best['model_state']) != set(embedded)
                or any(not torch.equal(best['model_state'][key], embedded[key]) for key in embedded)):
            raise ValueError('Best checkpoint does not reproduce committed validation selection')
        eligible.append(dict(config=config, status=status, checkpoint=checkpoint.relative_to(root).as_posix()))
    return eligible, excluded


def _binding(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = relative_path(root, path)
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing/nonlocal freeze artifact: {path}')
    return dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path), size_bytes=path.stat().st_size)


def _environment_paths(root):
    """Freeze the real lock/resolution inputs and bootstrap's installed receipts."""
    root = Path(root).resolve()
    required = {'environment/requirements-linux-lock.txt', 'environment/linux-resolve.json',
                'environment/candidate-status.json',
                'outputs/evidence/environment/installed-environment.json',
                'outputs/evidence/environment/pip-freeze.txt'}
    for relative in required:
        path = relative_path(root, relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing environment freeze input: {relative}')
    installed = read_json(root / 'outputs/evidence/environment/installed-environment.json')
    resolved = read_json(root / 'environment/linux-resolve.json')
    expected = {entry['metadata']['name']: entry['metadata']['version'] for entry in resolved.get('install', [])}
    if not expected or installed.get('linux_environment_verified') is not True or installed.get('packages') != expected:
        raise ValueError('Installed Linux environment receipt does not match the candidate resolution')
    paths = {relative_path(root, relative) for relative in required}
    paths.update(path for path in (root / 'environment').rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith(('.tmp', '.pyc')))
    return paths


def freeze_experiment(matrix, output, *, root=None):
    """Freeze already computed validation/calibration; never opens signal values."""
    root = _matrix_root(matrix, root)
    output = Path(output).resolve()
    if output.exists():
        verify_freeze(output, root=root)
        return output
    from training.a800_common import source_identity
    runs, excluded = _completed_runs(root, matrix)
    references = _historical_runs(root)
    selection_path = root / 'outputs/evidence/selection.json'
    selection = read_json(selection_path)
    paths = {Path(matrix).resolve(), selection_path}
    reports = []
    for entry in runs:
        config = entry['config']
        directory = root / 'outputs/runs' / config['run_id']
        validation_path = root / 'outputs/evidence/validation' / config['run_id'] / 'validation.json'
        calibration_path = root / 'outputs/evidence/calibration' / (config['run_id'] + '.json')
        validation, calibration_wrapper = read_json(validation_path), read_json(calibration_path)
        calibration = calibration_wrapper['result']
        _verify_validation_calibration(root, config, root / entry['checkpoint'], validation, calibration_wrapper)
        checked_model, _ = _load_model(root, config, root / entry['checkpoint'], 'cpu')
        del checked_model
        if calibration_wrapper.get('result_sha256') != _digest(calibration) or calibration['checkpoint_sha256'] != sha256_file(root / entry['checkpoint']):
            raise ValueError('Calibration was changed or uses another checkpoint')
        if validation['checkpoint_sha256'] != calibration['checkpoint_sha256'] or len(validation['conditions']) != 40:
            raise ValueError('Missing complete registered validation matrix')
        entry.update(calibration=calibration_path.relative_to(root).as_posix(), validation=validation_path.relative_to(root).as_posix())
        reports.append(dict(config=config, validation=validation))
        paths.update([root / entry['checkpoint'], directory / 'run-status.json', directory / 'last.pt', directory / 'data-access-audit.json', validation_path, calibration_path])
        paths.update([directory / 'best.pt.sha256.json', directory / 'last.pt.sha256.json'])
    if selection != select_experiment(_group_summary(reports)):
        raise ValueError('Selection does not reproduce registered validation gates')
    for entry in excluded:
        directory = root / 'outputs/runs' / entry['config']['run_id']
        paths.update([directory / 'run-status.json', directory / 'last.pt', directory / 'data-access-audit.json'])
        paths.add(directory / 'last.pt.sha256.json')
    for entry in references:
        config = entry['config']
        validation_path = root / 'outputs/evidence/validation' / config['run_id'] / 'validation.json'
        calibration_path = root / 'outputs/evidence/calibration' / (config['run_id'] + '.json')
        calibration = read_json(calibration_path)
        validation = read_json(validation_path)
        _verify_validation_calibration(root, config, root / entry['checkpoint'], validation, calibration)
        if calibration.get('result_sha256') != _digest(calibration.get('result')) or calibration['result']['checkpoint_sha256'] != sha256_file(root / entry['checkpoint']) or validation['checkpoint_sha256'] != calibration['result']['checkpoint_sha256'] or len(validation['conditions']) != 40:
            raise ValueError('Historical reference validation/calibration is incomplete or changed')
        entry.update(calibration=calibration_path.relative_to(root).as_posix(), validation=validation_path.relative_to(root).as_posix())
        paths.update([root / entry['checkpoint'], root / entry['protocol_path'], validation_path, calibration_path])
        paths.update(root / entry['reference'][key] for key in ('training_evidence', 'resolved_config'))
    for entry in runs + excluded:
        audit = read_json(root / 'outputs/runs' / entry['config']['run_id'] / 'data-access-audit.json')
        if audit.get('sealed_access_count') != 0 or not set(audit.get('splits', [])).issubset({'train', 'validation'}) or not audit.get('splits'):
            raise ValueError('Training data-access audit is missing or admits sealed data')
    # Metadata, arrays, and source manifests are all bound; original ZIP files
    # may contain sealed members, but only their bytes are hashed at this point.
    for config in read_json(matrix)['runs']:
        for data_root in (config['data']['train_root'], config['data']['validation_root']):
            base = relative_path(root, data_root)
            paths.add(base / 'dataset-metadata.json')
            for split in ('train', 'validation'):
                if (base / split).exists():
                    paths.update(base / split / name for name in ('data.npy', 'labels.npy', 'windows.csv'))
    sealed_path = root / 'manifests/sealed-evaluation.json'
    sealed = read_json(sealed_path)
    if (len(sealed.get('known_test', [])) != 24 or len(sealed.get('unknown', [])) != 204
            or len({r['archive_id'] for r in sealed.get('unknown', [])}) != 4):
        raise ValueError('Sealed manifest requires 24 known and 204 unknown members from four unknown sources')
    all_members = sealed['known_test'] + sealed['unknown']
    if len({r['recording_id'] for r in all_members}) != len(all_members):
        raise ValueError('Duplicated sealed member identity')
    for collection, split in (('known_test', 'test'), ('unknown', 'unknown')):
        for member in sealed[collection]:
            if (member['split'] != split or len(member.get('window_starts', [])) != 32
                    or len(set(member.get('window_ids', []))) != 32
                    or (member['label_index'] not in (0, 1, 2) if split == 'test' else member['label_index'] != -1)):
                raise ValueError('Sealed member window or label contract mismatch')
    paths.add(sealed_path)
    for item in sealed['sources'] + sealed.get('manifest_files', []):
        path = relative_path(root, item['path'])
        if sha256_file(path) != item['sha256'] or path.stat().st_size != item['size_bytes']:
            raise ValueError('Sealed source/manifest identity mismatch')
        paths.add(path)
    vti_path = root / 'manifests/vti-audit.json'
    if vti_path.is_file():
        paths.add(vti_path)
    paths.update(relative_path(root, relative) for relative in source_identity(root))
    paths.update(p for p in (root / 'training/configs/a800_v2').glob('*.json'))
    environment_paths = _environment_paths(root)
    paths.update(environment_paths)
    payload = dict(schema_version='2.0', artifact_type='a800_v2_frozen_experiment', root_relative=os.path.relpath(root, output.parent).replace('\\', '/'),
                   inference_precision='fp32', primary_score='cosine', comparison='score>=threshold', deployment_seed=20260909,
                   selection=selection, conditions=conditions(), runs=runs, historical_references=references, excluded=excluded, label_map=LABEL_MAP,
                   sealed_manifest='manifests/sealed-evaluation.json', vti_audit='manifests/vti-audit.json' if vti_path.exists() else None,
                   environment_files=sorted(p.relative_to(root).as_posix() for p in environment_paths),
                   bindings=[_binding(root, p) for p in sorted(paths)],
                   limitations=['Three registered known RF control sources; not general UAV presence detection.',
                                'MAT member boundaries do not establish independent physical acquisition sessions.',
                                'Synthetic added-noise ratios are not physical acquisition SNR.',
                                'Same-user process checks are workflow isolation, not an adversarial security boundary.'])
    payload['content_sha256'] = _digest(payload)
    atomic_json(output, payload)
    verify_freeze(output, root=root)
    return output


class _MemberDataset:
    def __init__(self, member, windows):
        self.member, self.data = member, windows
        self.split = member['split']
        if (np.asarray(windows).shape != (32, 2, 4096) or len(member.get('window_ids', [])) != 32
                or len(set(member['window_ids'])) != 32 or len(member.get('window_starts', [])) != 32):
            raise ValueError('Sealed members require all 32 frozen windows and original window IDs')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index], dtype=np.float32, copy=True)), int(self.member['label_index']), index

    def sample_metadata(self, index):
        start = self.member['window_starts'][index]
        identifiers = self.member['window_ids']
        return dict(recording_id=self.member['recording_id'], start_sample=start,
                    window_id=identifiers[index])


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _open_bootstrap(known_rows, unknown_rows, score, threshold, allocation=None):
    groups = []
    rows = known_rows + unknown_rows
    strata = [f'known:{r["label"]}' for r in known_rows] + [f'unknown:{r["archive_id"]}' for r in unknown_rows]
    for key in sorted(set(strata)):
        groups.append(np.flatnonzero(np.asarray(strata) == key))
    generator = np.random.default_rng(2026090904)
    values = {key: [] for key in ('auroc', 'fpr95', 'known_acceptance', 'unknown_false_acceptance')}
    for iteration in range(2000):
        if allocation is not None and iteration % 50 == 0:
            check_allocation(allocation)
        ix = np.concatenate([generator.choice(g, len(g), replace=True) for g in groups])
        known = [rows[i][score] for i in ix if i < len(known_rows)]
        unknown = [rows[i][score] for i in ix if i >= len(known_rows)]
        result = open_set_metrics(known, unknown, threshold)
        for key in values:
            values[key].append(result[key])
    return dict(ci95={key: np.quantile(val, [.025, .975]).tolist() for key, val in values.items()},
                replicates=2000, seed=2026090904, unit='known-class/unknown-source stratified MAT member',
                limitation='Physical session correlation remains unknown')


def evaluate_frozen(freeze, *, output_dir, allocation=None):
    start_finalization(allocation)
    payload = verify_freeze(freeze)
    root, output_dir = Path(payload['_root']), Path(output_dir).resolve()
    if root != allocation.root.resolve():
        raise ValueError('Freeze and allocation refer to different experiment roots')
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / 'evaluation-completed.json'
    if completion.exists():
        result = read_json(completion)
        if result['freeze_sha256'] != sha256_file(freeze):
            raise ValueError('Completed evaluation belongs to another freeze')
        for item in result['artifacts']:
            if sha256_file(relative_path(output_dir, item['path'])) != item['sha256']:
                raise ValueError('Completed evaluation result changed')
        return output_dir / 'metrics.json'
    from training.a800_data import load_sealed_member
    sealed = read_json(root / payload['sealed_manifest'])
    identity = sha256_file(freeze)
    all_metrics, all_members, all_windows, robustness, open_curves = [], [], [], [], []
    clean_by_run = {}
    for run in payload['runs'] + payload.get('historical_references', []):
        check_allocation(allocation)
        config = run['config']
        model, _ = _load_model(root, config, root / run['checkpoint'], 'cuda', allocation=allocation)
        calibration = read_json(root / run['calibration'])['result']
        prototype = np.asarray(calibration['prototypes'])
        known_by_condition, unknown_rows = {}, []
        for collection in ('known_test', 'unknown'):
            members = sealed[collection]
            condition_list = payload['conditions'] if collection == 'known_test' else payload['conditions'][:1]
            for member in members:
                check_allocation(allocation)
                # Values are opened only after complete freeze verification above.
                loaded = None
                for condition in condition_list:
                    key = _digest(dict(run=config['run_id'], member=member['recording_id'], condition=condition['id']))
                    path = output_dir / 'shards' / (key + '.json')
                    def compute():
                        nonlocal loaded
                        if loaded is None:
                            loaded = load_sealed_member(root, member)
                        dataset = _MemberDataset(member, loaded)
                        if condition['spec'] is not None:
                            dataset = DeterministicIQPerturbationDataset(dataset, perturbation=condition['spec'], base_seed=condition['repeat_seed'])
                        result = predict_dataset(model, dataset, 'cuda', prototypes=prototype, allocation=allocation)
                        row = result['members'][0]
                        row.update(archive_id=member['archive_id'], split=member['split'], run_id=config['run_id'], condition_id=condition['id'])
                        row.pop('feature')
                        for window in result['windows']:
                            window.update(run_id=config['run_id'], condition_id=condition['id'], split=member['split'])
                        return dict(member=row, windows=result['windows'])
                    result = cached_shard(path, identity, compute)
                    all_members.append(result['member'])
                    all_windows.extend(result['windows'])
                    if collection == 'known_test':
                        known_by_condition.setdefault(condition['id'], []).append(result['member'])
                    else:
                        unknown_rows.append(result['member'])
        for condition in payload['conditions']:
            metrics = classification(known_by_condition[condition['id']])
            metrics['bootstrap'] = classification_bootstrap(known_by_condition[condition['id']])
            row = dict(run_id=config['run_id'], experiment=config['experiment'], seed=config['seed'], condition=condition, **metrics)
            all_metrics.append(row)
            robustness.append(row)
        clean_rows = known_by_condition['clean']
        clean_by_run[config['run_id']] = clean_rows
        for score in SCORES:
            threshold = calibration['scores'][score]['threshold']
            report = open_set_metrics([r[score] for r in clean_rows], [r[score] for r in unknown_rows], threshold)
            accepted = [r for r in clean_rows if r[score] >= threshold]
            report.update(known_classification_coverage=len(accepted) / len(clean_rows),
                          correct_and_accepted_fraction=sum(r['prediction'] == r['label'] for r in accepted) / len(clean_rows),
                          accepted_classification_accuracy=None if not accepted else sum(r['prediction'] == r['label'] for r in accepted) / len(accepted))
            bootstrap = cached_shard(output_dir / 'shards' / f'bootstrap-{config["run_id"]}-{score}.json', identity,
                                     lambda: _open_bootstrap(clean_rows, unknown_rows, score, threshold, allocation))
            open_curves.append(dict(run_id=config['run_id'], score=score, primary=score == 'cosine', **report, bootstrap=bootstrap,
                                    source_metrics={source: open_set_metrics([r[score] for r in clean_rows], [r[score] for r in unknown_rows if r['archive_id'] == source], threshold) for source in sorted({r['archive_id'] for r in unknown_rows})}))
        del model
        torch.cuda.empty_cache()
    comparisons = []
    for run in payload['runs']:
        config = run['config']
        if config['experiment'] == 'A':
            continue
        base = sorted(clean_by_run[f'A-seed{config["seed"]}'], key=lambda r: r['recording_id'])
        candidate = sorted(clean_by_run[config['run_id']], key=lambda r: r['recording_id'])
        if [r['recording_id'] for r in base] != [r['recording_id'] for r in candidate]:
            raise ValueError('Paired test members do not align')
        comparisons.append(dict(run_id=config['run_id'], comparison=f'A-seed{config["seed"]}',
            **paired_bootstrap([r['label'] for r in base], [r['prediction'] for r in base], [r['prediction'] for r in candidate], allocation=allocation)))
    summaries = []
    for experiment in ['A', 'B', 'C', 'D', 'V4']:
        for condition in payload['conditions']:
            entries = [r for r in all_metrics if r['experiment'] == experiment and r['condition']['id'] == condition['id']]
            if not entries:
                continue
            scores = [r['macro_f1'] for r in entries]
            summaries.append(dict(experiment=experiment, condition_id=condition['id'], seeds=[r['seed'] for r in entries],
                                  macro_f1_mean=fmean(scores), macro_f1_sample_std=stdev(scores) if len(scores) > 1 else None,
                                  seeds_are_not_independent_recordings=True))
    vti = read_json(root / payload['vti_audit']) if payload.get('vti_audit') else {'status': 'missing_audit'}
    vti_result = dict(status='blocked_incompatible_representation', audit=vti,
                      reason='No separately frozen semantically compatible VTI member adapter and protocol is registered; no VTI predictions made.')
    result = dict(schema_version='2.0', freeze_sha256=identity, metrics=all_metrics, seed_summaries=summaries,
                  open_set=open_curves, paired_clean_comparisons=comparisons, primary_unit='MAT member',
                  primary_open_set_score='cosine', vti=vti_result, exclusions=payload['excluded'], limitations=payload['limitations'])
    atomic_json(output_dir / 'metrics.json', result)
    _write_csv(output_dir / 'metrics.csv', [{**r, 'condition_id': r['condition']['id']} for r in all_metrics])
    _write_csv(output_dir / 'member-predictions.csv', all_members)
    _write_csv(output_dir / 'window-predictions.csv', all_windows)
    atomic_json(output_dir / 'confusion.json', {'matrices': [{k: r[k] for k in ('run_id', 'condition', 'confusion_matrix')} for r in all_metrics]})
    atomic_json(output_dir / 'robustness-curves.json', {'points': robustness})
    atomic_json(output_dir / 'open-set-curves.json', {'curves': open_curves})
    atomic_json(output_dir / 'failure-cases.json', {'known_errors': [r for r in all_members if r['label'] >= 0 and r['label'] != r['prediction']]})
    artifacts = [dict(path=p.relative_to(output_dir).as_posix(), sha256=sha256_file(p)) for p in sorted(output_dir.rglob('*')) if p.is_file() and p.name != 'evaluation-completed.json' and not p.name.endswith('.tmp')]
    atomic_json(completion, dict(freeze_sha256=identity, artifacts=artifacts, status='completed'))
    return output_dir / 'metrics.json'


def run_finalization(root=None, *, allocation=None, matrix_path=None, allocation_confirmed=False, session_minutes=None, gpu_index=0):
    from training.a800_common import require_allocation
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    matrix = Path(matrix_path or root / 'training/configs/a800_v2/matrix.json').resolve()
    if allocation is None:
        allocation = require_allocation(root, gpu_index=gpu_index, session_minutes=session_minutes,
                                        allocation_confirmed=allocation_confirmed, phase='finalize')
    start_finalization(allocation)
    if root != allocation.root.resolve():
        raise ValueError('Finalization and allocation refer to different roots')
    freeze = root / 'outputs/evidence/freeze.json'
    if not freeze.exists():
        runs, _ = _completed_runs(root, matrix)
        reports = []
        for run in runs + _historical_runs(root):
            config = run['config']
            validation = validate_run(config, root / run['checkpoint'], root=root, allocation=allocation)
            calibrate_run(config, root / run['checkpoint'], root=root, allocation=allocation)
            if config['kind'] == 'formal':
                reports.append(dict(config=config, validation=validation))
        atomic_json(root / 'outputs/evidence/selection.json', select_experiment(_group_summary(reports)))
        freeze_experiment(matrix, freeze, root=root)
    results = evaluate_frozen(freeze, output_dir=root / 'outputs/evidence/test', allocation=allocation)
    from training.a800_delivery import benchmark_frozen, export_release
    benchmark = benchmark_frozen(freeze, allocation=allocation)
    release = export_release(freeze, results, root / 'outputs/release', allocation=allocation)
    return dict(status='completed', freeze=str(freeze), results=str(results), benchmark=str(benchmark), release=str(release))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--session-minutes', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(argv)
    from training.a800_common import AllocationUnavailable
    try:
        result = run_finalization(args.root, matrix_path=args.matrix, allocation_confirmed=args.allocation_confirmed,
                                  session_minutes=args.session_minutes, gpu_index=args.gpu)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (InterruptedError, AllocationUnavailable) as exc:
        print(json.dumps(dict(status='interrupted', reason=str(exc))))
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
