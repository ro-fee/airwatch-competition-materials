"""Fixed A800 V2 matrix, atomic complete-state recovery and finite GPU sessions.

Formal runs are CUDA/BF16 only and require a verified Allocation. The reusable
loop also accepts explicitly named engineering-smoke fixtures on CPU. No test or
unknown dataset is accepted by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.a800_common import (Allocation, AllocationUnavailable, atomic_json,
    bundle_path, canonical_hash, require_allocation, sha256_file, source_identity)


PROTOCOL = 'airwatch-a800-v2'
SEEDS = (20260909, 20260910, 20260911)
EXPERIMENTS = ('A', 'B', 'C', 'D')


def make_run_config(experiment: str, seed: int) -> dict:
    if experiment not in EXPERIMENTS or seed not in SEEDS:
        raise ValueError('experiment/seed is outside the preregistered matrix')
    return {'schema_version': '2.0', 'protocol': PROTOCOL,
        'run_id': f'{experiment}-seed{seed}', 'kind': 'formal',
        'experiment': experiment, 'seed': seed,
        'model': {'name': 'tcn' if experiment in 'AB' else 'multiscale_tcn', 'num_classes': 3},
        'data': {'train_root': 'data/prepared/known-iq-v1' if experiment == 'A' else 'data/prepared/dense-train-v2',
                 'validation_root': 'data/prepared/known-iq-v1',
                 'train_artifact': 'v1' if experiment == 'A' else 'v2'},
        'training': {'anchors_per_batch': 128, 'steps_per_epoch': 230,
            'logical_epochs': 40, 'total_steps': 9200,
            'lr': .001, 'min_lr': .00001, 'warmup_steps': 230,
            'weight_decay': .0001, 'betas': [.9, .999], 'eps': 1e-8,
            'gradient_clip_norm': 1.0, 'precision': 'bf16', 'workers': 4,
            'cpu_threads': 4, 'interop_threads': 1,
            'js_max_weight': .1 if experiment == 'D' else 0., 'js_warmup_steps': 230},
        'selection': {'clean_f1_floor': .97, 'repeat_seeds': [2026090901, 2026090902, 2026090903]},
        'augmentation_protocol': 'v4-paired-position-v1'}


def validate_config(config: dict) -> None:
    if config.get('kind') == 'formal':
        expected = make_run_config(config.get('experiment'), config.get('seed'))
        if config != expected:
            raise ValueError('formal config/budget differs from the frozen matrix; a new protocol is required')
    elif config.get('kind') == 'engineering_smoke':
        if not config.get('run_id', '').startswith('engineering-smoke-'):
            raise ValueError('engineering smoke requires a separate engineering-smoke- run ID')
        if config.get('protocol') != PROTOCOL:
            raise ValueError('unsupported smoke protocol')
        training = config['training']
        if not 1 <= training['total_steps'] <= 200:
            raise ValueError('engineering smoke budget must be between 1 and 200 steps')
        if training['precision'] not in {'fp32', 'bf16'}:
            raise ValueError('unsupported smoke precision')
    else:
        raise ValueError('run kind must be formal or engineering_smoke')
    training = config['training']
    for name in ('anchors_per_batch', 'steps_per_epoch', 'logical_epochs', 'total_steps', 'warmup_steps'):
        if type(training[name]) is not int or training[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if training['steps_per_epoch'] * training['logical_epochs'] != training['total_steps']:
        raise ValueError('logical epoch and total budget mismatch')
    if not 0 < training['min_lr'] <= training['lr'] or training['warmup_steps'] > training['total_steps']:
        raise ValueError('invalid learning-rate schedule')
    for path in config['data'].values():
        if path not in {'v1', 'v2'} and ('sealed' in Path(path).parts or Path(path).is_absolute()):
            raise ValueError('training only accepts portable train/validation paths')


def load_matrix(root: Path) -> list[dict]:
    payload = json.loads((Path(root)/'training/configs/a800_v2/matrix.json').read_text(encoding='utf-8'))
    expected = [make_run_config(group, seed) for group in EXPERIMENTS for seed in SEEDS]
    if payload != {'schema_version': '2.0', 'protocol': PROTOCOL, 'runs': expected}:
        raise ValueError('formal matrix does not match the preregistered 12-run protocol')
    return payload['runs']


def epoch_order(size: int, draws: int, *, seed: int, epoch: int) -> np.ndarray:
    if size <= 0 or draws <= 0 or draws % size:
        raise ValueError('each logical epoch must contain complete independent permutations')
    orders = []
    for cycle in range(draws // size):
        digest = hashlib.sha256(f'a800-order-v1|{seed}|{epoch}|{cycle}'.encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], 'little')))
        orders.append(rng.permutation(size))
    return np.concatenate(orders).astype(np.int64)


def configure_determinism(seed: int, training: dict, device: torch.device) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = str(training['cpu_threads'])
    os.environ['MKL_NUM_THREADS'] = str(training['cpu_threads'])
    torch.set_num_threads(training['cpu_threads'])
    if torch.get_num_interop_threads() != training.get('interop_threads', 1):
        torch.set_num_interop_threads(training.get('interop_threads', 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def _restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


CHECKPOINT_FIELDS = {'schema_version', 'artifact_type', 'config', 'identity', 'model_state',
    'optimizer_state', 'scheduler_state', 'scaler', 'global_step', 'logical_epoch', 'epoch_step',
    'draw_position', 'sampler', 'rng', 'best_rank', 'best_step', 'best_state', 'history',
    'kind', 'validation_pending', 'timing'}


def load_checkpoint(path: Path, *, expected_identity: dict | None = None) -> dict:
    """Read an internally generated checkpoint only after verifying its commit hash."""
    path = Path(path)
    try:
        receipt = json.loads(path.with_suffix(path.suffix + '.sha256.json').read_text(encoding='utf-8'))
        if receipt['sha256'] != sha256_file(path) or receipt['size_bytes'] != path.stat().st_size:
            raise ValueError('checkpoint hash mismatch')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(value, dict) or not CHECKPOINT_FIELDS.issubset(value):
            raise ValueError('incomplete checkpoint')
        if value['schema_version'] != '2.0' or value['artifact_type'] != 'a800_training_checkpoint':
            raise ValueError('unsupported checkpoint schema')
        if value['scaler'] is not None:
            raise ValueError('BF16/FP32 checkpoint must not contain a GradScaler')
        training = value['config']['training']
        step = value['global_step']
        if not 0 <= step <= training['total_steps'] or value['draw_position'] != step * training['anchors_per_batch']:
            raise ValueError('incomplete checkpoint cursor')
        epoch, epoch_step = divmod(step, training['steps_per_epoch'])
        if (value['logical_epoch'] != epoch or value['epoch_step'] != epoch_step
                or value['sampler'].get('epoch') != epoch
                or value['sampler'].get('position') != epoch_step * training['anchors_per_batch']):
            raise ValueError('checkpoint sampler cursor is inconsistent')
        if value['identity']['config_sha256'] != canonical_hash(value['config']):
            raise ValueError('checkpoint config identity mismatch')
        if set(value['rng']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
            raise ValueError('checkpoint RNG state is incomplete')
        if expected_identity is not None and value['identity'] != expected_identity:
            raise ValueError('resume identity mismatch: configuration, data, source or precision changed')
        return value
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f'incomplete checkpoint: {path.name}: {exc}') from exc


def _save_checkpoint(path: Path, state: dict, *, rotate: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {'sha256': sha256_file(temporary), 'size_bytes': temporary.stat().st_size}
        if rotate and path.exists():
            try:
                load_checkpoint(path)
            except Exception:
                pass  # Keep the previous valid recovery point when latest is corrupt.
            else:
                previous = path.with_name('last.previous.pt')
                os.replace(path, previous)
                os.replace(path.with_suffix('.pt.sha256.json'), previous.with_suffix('.pt.sha256.json'))
        os.replace(temporary, path)
        atomic_json(path.with_suffix('.pt.sha256.json'), receipt)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_identity(dataset) -> dict:
    identity = copy.deepcopy(dataset.data_identity)
    paths = {name: getattr(dataset, name, None) for name in ('data_path', 'labels_path', 'windows_path')}
    if hasattr(dataset, 'root'):
        paths['metadata'] = Path(dataset.root) / 'dataset-metadata.json'
        for key, filename in (('data_path', 'data.npy'), ('labels_path', 'labels.npy'), ('windows_path', 'windows.csv')):
            if paths[key] is None:
                paths[key] = Path(dataset.root) / dataset.split / filename
    identity['artifacts'] = {name: sha256_file(Path(path)) for name, path in paths.items() if path is not None}
    return identity


class _PairedEpoch(Dataset):
    """Prefetch is speculative; only the main optimizer loop advances draw_position."""
    def __init__(self, dataset, order, seed, epoch, offset):
        self.dataset, self.order, self.seed, self.epoch, self.offset = dataset, order, seed, epoch, offset

    def __len__(self):
        return len(self.order) - self.offset

    def __getitem__(self, item):
        from training.a800_objective import make_paired_views
        position = self.offset + item
        index = int(self.order[position])
        values, label, _ = self.dataset[index]
        window_id = self.dataset.sample_metadata(index).window_id
        first, second = make_paired_views(values.unsqueeze(0), [window_id], run_seed=self.seed,
            logical_epoch=self.epoch, draw_positions=[position])
        return first[0], second[0], label, index, position


def _default_model(config):
    from airwatch.models.uav_multiscale_tcn import build_a800_model
    return build_a800_model(config['model']['name'], num_classes=config['model']['num_classes'])


def _verify_gpu_preflight(root, allocation, code):
    path = root/'outputs/evidence/gpu-preflight.json'
    if not path.is_file():
        raise ValueError('formal run requires completed allocated CUDA preflight')
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('state') != 'completed' or report.get('code') != code
            or report.get('gpu_uuid') != allocation.gpu_uuid
            or {row['model'] for row in report.get('models', [])} != {'tcn', 'multiscale_tcn'}
            or any(row.get('resume_equivalence', {}).get('exact_equal') is not True
                   or row['resume_equivalence'].get('state') != 'completed'
                   for row in report.get('models', []))):
        raise ValueError('CUDA preflight failed, is incomplete or has changed identity')


def run_training(config: dict, train_dataset, validation_dataset, output_dir: Path, *,
                 device='cuda', allocation: Allocation | None = None, code_identity: dict | None = None,
                 model_factory=None, evaluator=None, stop_after_steps: int | None = None,
                 stop_requested=None) -> dict:
    """Train/resume one fixed run; injected models/evaluators are for engineering fixtures."""
    validate_config(config)
    formal = config['kind'] == 'formal'
    device = torch.device(device)
    if formal and (allocation is None or device.type != 'cuda'):
        raise ValueError('formal training requires verified allocation and CUDA')
    if formal and (model_factory is not None or evaluator is not None or stop_after_steps is not None):
        raise ValueError('formal training cannot replace the model, evaluator or budget')
    if train_dataset.split != 'train' or (formal and validation_dataset.split != 'validation'):
        raise ValueError('training refuses sealed/test/unknown inputs')
    if formal:
        allocation.recheck()
        if allocation.phase != 'train':
            raise ValueError('formal training requires a train-phase allocation')
        if len(train_dataset) != (3680 if config['experiment'] == 'A' else 29440) or len(validation_dataset) != 768:
            raise ValueError('formal dataset coverage differs from frozen protocol')
        if train_dataset.recording_ids & validation_dataset.recording_ids:
            raise ValueError('train/validation recording leakage')
    elif device.type == 'cuda':
        if allocation is None:
            raise ValueError('GPU engineering smoke also requires explicit allocation')
        allocation.recheck()
    output_dir = Path(output_dir)
    root = allocation.root if allocation is not None else Path(__file__).resolve().parents[1]
    if formal:
        current_code = source_identity(root)
        if code_identity is not None and code_identity != current_code:
            raise ValueError('formal source identity must match the immutable bundle')
        code_identity = current_code
        _verify_gpu_preflight(root, allocation, current_code)
    identity = {'config_sha256': canonical_hash(config),
                'data': {'train': _dataset_identity(train_dataset), 'validation': _dataset_identity(validation_dataset)},
                'code': code_identity if code_identity is not None else source_identity(root),
                'precision': config['training']['precision'],
                'environment': {'python': sys.version.split()[0], 'numpy': np.__version__,
                    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version() if device.type == 'cuda' else None,
                    'device_type': device.type}}
    last_path = output_dir / 'last.pt'
    resume = None
    if last_path.exists() or (output_dir / 'last.previous.pt').exists():
        try:
            resume = load_checkpoint(last_path)
        except Exception:
            resume = load_checkpoint(output_dir / 'last.previous.pt')
        if resume['identity'] != identity:
            raise ValueError('resume identity mismatch: configuration, data, source or precision changed')
    if (output_dir / 'run-status.json').exists():
        old_status = json.loads((output_dir / 'run-status.json').read_text(encoding='utf-8'))
        if old_status['state'] == 'failed' and old_status.get('reason') != 'failed_no_eligible_checkpoint':
            raise ValueError('failed run requires diagnosis; automatic retry is not permitted')
        if old_status['state'] == 'completed' and resume is None:
            raise ValueError('completed status has no valid checkpoint')
    training = config['training']
    configure_determinism(config['seed'], training, device)
    if device.type == 'cuda' and (not torch.cuda.is_bf16_supported() or training['precision'] != 'bf16'):
        raise ValueError('allocated CUDA device must support the frozen BF16 protocol')
    model = (model_factory or _default_model)(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['lr'],
        betas=tuple(training['betas']), eps=training['eps'], weight_decay=training['weight_decay'])
    def lr_factor(step):
        if step < training['warmup_steps']:
            return (step + 1) / training['warmup_steps']
        span = max(1, training['total_steps'] - training['warmup_steps'] - 1)
        progress = min(1., (step - training['warmup_steps']) / span)
        minimum = training['min_lr'] / training['lr']
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    classes = config['model']['num_classes']
    counts = torch.bincount(torch.tensor([sample.label_index for sample in train_dataset.samples]), minlength=classes)
    if len(counts) != classes or (counts <= 0).any():
        raise ValueError('each known training class requires real samples')
    class_weights = counts.float().reciprocal()
    class_weights = (class_weights / class_weights.mean()).to(device)
    state = {'schema_version': '2.0', 'artifact_type': 'a800_training_checkpoint',
        'config': copy.deepcopy(config), 'identity': identity, 'kind': config['kind'],
        'global_step': 0, 'logical_epoch': 0, 'epoch_step': 0, 'draw_position': 0,
        'best_rank': None, 'best_step': None, 'best_state': None, 'history': [],
        'validation_pending': False, 'scaler': None, 'sampler': {},
        'timing': {'elapsed_compute_seconds': 0., 'gpu_step_milliseconds': []}}
    if resume is not None:
        state = resume
        saved_epoch = state['global_step'] // training['steps_per_epoch']
        saved_order = epoch_order(len(train_dataset), training['anchors_per_batch'] * training['steps_per_epoch'],
                                  seed=config['seed'], epoch=saved_epoch)
        if state['sampler']['order_sha256'] != hashlib.sha256(saved_order.tobytes()).hexdigest():
            raise ValueError('checkpoint sampler plan identity mismatch')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        _restore_rng(state['rng'], device)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'config.json', config)
    atomic_json(output_dir / 'data-access-audit.json', {'kind': config['kind'],
        'allowed_splits': ['train', 'validation'], 'splits': ['train', 'validation'],
        'sealed_access_count': 0, 'test_accessed': False, 'unknown_accessed': False,
        'inputs': identity['data'], 'config_sha256': identity['config_sha256']})
    batch_size = training['anchors_per_batch']
    draws = batch_size * training['steps_per_epoch']
    if draws % len(train_dataset):
        raise ValueError('training epoch exposure must contain full permutations')
    if evaluator is None:
        from training.a800_evaluation import evaluate_validation
        evaluator = evaluate_validation
    def stopping():
        return ((allocation is not None and allocation.should_stop())
                or (stop_after_steps is not None and state['global_step'] >= stop_after_steps)
                or (stop_requested is not None and stop_requested()))
    session_started = time.perf_counter()
    previous_elapsed = state['timing']['elapsed_compute_seconds']
    def save():
        state['timing']['elapsed_compute_seconds'] = previous_elapsed + time.perf_counter() - session_started
        epoch, cursor = divmod(state['global_step'], training['steps_per_epoch'])
        order = epoch_order(len(train_dataset), draws, seed=config['seed'], epoch=epoch)
        state.update(model_state=_cpu_copy(model.state_dict()), optimizer_state=_cpu_copy(optimizer.state_dict()),
                     scheduler_state=copy.deepcopy(scheduler.state_dict()), rng=_rng_state(device),
                     logical_epoch=epoch, epoch_step=cursor,
                     sampler={'epoch': epoch, 'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                              'position': cursor * batch_size})
        _save_checkpoint(last_path, state, rotate=True)
        atomic_json(output_dir/'timing.json', state['timing'])
        if state['best_state'] is not None:
            best_path = output_dir / 'best.pt'
            # Embedded best state makes recovery independent of a partially written best.pt.
            try:
                stored_best = load_checkpoint(best_path)
                best_is_current = stored_best['global_step'] == state['best_step']
            except Exception:
                best_is_current = False
            if not best_is_current:
                best = dict(state, model_state=state['best_state'], global_step=state['best_step'],
                            draw_position=state['best_step'] * batch_size)
                best.update(state['best_training_state'])
                _save_checkpoint(best_path, best, rotate=False)
    def status(name, reason=None):
        result = {'schema_version': '2.0', 'run_id': config['run_id'], 'state': name,
            'reason': reason, 'kind': config['kind'], 'experiment': config['experiment'], 'seed': config['seed'],
            'global_step': state['global_step'], 'total_steps': training['total_steps'],
            'anchors_presented': state['draw_position'], 'forward_inputs_presented': 2 * state['draw_position'],
            'elapsed_compute_seconds': state['timing']['elapsed_compute_seconds'],
            'measured_gpu_step_seconds': sum(state['timing']['gpu_step_milliseconds']) / 1000,
            'config_sha256': identity['config_sha256'],
            'checkpoint_sha256': sha256_file(last_path) if last_path.exists() else None,
            'best_checkpoint_sha256': sha256_file(output_dir/'best.pt') if (output_dir/'best.pt').exists() else None}
        atomic_json(output_dir / 'run-status.json', result)
        return result
    status('running')
    try:
        while state['global_step'] < training['total_steps'] or state['validation_pending']:
            if stopping():
                save()
                return status('interrupted', 'allocation_boundary_or_stop_requested')
            if state['validation_pending']:
                model.eval()
                # Main evaluation is always FP32; no enclosing autocast is active.
                metrics = evaluator(model, validation_dataset, device, batch_size=64, allocation=allocation)
                rank = metrics['rank']
                if rank is not None and (len(rank) != 4 or not all(math.isfinite(float(value)) for value in rank)):
                    raise ValueError('non-finite or incomplete validation rank')
                if rank is not None and (state['best_rank'] is None or tuple(rank) > tuple(state['best_rank'])):
                    state['best_rank'], state['best_step'] = list(rank), state['global_step']
                    state['best_state'] = _cpu_copy(model.state_dict())
                    best_epoch, best_cursor = divmod(state['global_step'], training['steps_per_epoch'])
                    best_order = epoch_order(len(train_dataset), draws, seed=config['seed'], epoch=best_epoch)
                    state['best_training_state'] = {
                        'optimizer_state': _cpu_copy(optimizer.state_dict()),
                        'scheduler_state': copy.deepcopy(scheduler.state_dict()), 'rng': _rng_state(device),
                        'logical_epoch': best_epoch, 'epoch_step': best_cursor, 'validation_pending': False,
                        'sampler': {'epoch': best_epoch, 'position': best_cursor * batch_size,
                            'order_sha256': hashlib.sha256(best_order.tobytes()).hexdigest()}}
                state['history'].append({'event': 'validation', 'global_step': state['global_step'], 'metrics': metrics})
                state['validation_pending'] = False
                save()
                status('running')
                print(json.dumps({'run_id': config['run_id'], 'step': state['global_step'], 'event': 'validation', 'rank': rank}), flush=True)
                continue
            epoch, epoch_step = divmod(state['global_step'], training['steps_per_epoch'])
            order = epoch_order(len(train_dataset), draws, seed=config['seed'], epoch=epoch)
            paired = _PairedEpoch(train_dataset, order, config['seed'], epoch, epoch_step * batch_size)
            generator = torch.Generator().manual_seed(config['seed'] + epoch)
            loader = DataLoader(paired, batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=training['workers'], pin_memory=device.type == 'cuda', generator=generator)
            for first, second, labels, indices, positions in loader:
                if stopping():
                    save()
                    return status('interrupted', 'allocation_boundary_or_stop_requested')
                model.train()
                view_hash = hashlib.sha256(first.numpy().tobytes() + second.numpy().tobytes()).hexdigest()
                first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                gpu_start = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                gpu_end = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                if gpu_start is not None:
                    gpu_start.record()
                coefficient = training['js_max_weight'] * min(1., state['global_step'] / training['js_warmup_steps'])
                autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                from training.a800_objective import paired_objective
                lr = optimizer.param_groups[0]['lr']
                with autocast:
                    loss, diagnostics = paired_objective(model, first, second, labels,
                        js_weight=coefficient, class_weights=class_weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError('NaN/Inf loss; the batch was not skipped')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                if gpu_end is not None:
                    gpu_end.record()
                    gpu_end.synchronize()
                    state['timing']['gpu_step_milliseconds'].append(gpu_start.elapsed_time(gpu_end))
                if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError('NaN/Inf parameter after optimizer step')
                scheduler.step()
                state['global_step'] += 1
                state['draw_position'] += len(indices)
                state['history'].append({'event': 'step', 'global_step': state['global_step'],
                    'indices': indices.tolist(), 'positions': positions.tolist(), 'views_sha256': view_hash,
                    'lr': lr, 'js_weight': coefficient,
                    'loss': {key: float(value) for key, value in diagnostics.items()}})
                if state['global_step'] % 50 == 0:
                    print(json.dumps({'run_id': config['run_id'], 'step': state['global_step'],
                                      'loss': float(loss.detach()), 'lr': lr}), flush=True)
                if state['global_step'] % training['steps_per_epoch'] == 0:
                    state['validation_pending'] = True
                    break
        save()
        if state['best_state'] is None:
            return status('failed', 'failed_no_eligible_checkpoint')
        return status('completed')
    except (InterruptedError, AllocationUnavailable):
        save()
        return status('interrupted', 'allocation_boundary_during_validation')
    except Exception as exc:
        # Never overwrite the last valid recovery point with a failed partial update.
        status('failed', f'{type(exc).__name__}: {exc}')
        atomic_json(output_dir / 'failure.json', {'global_step': state['global_step'],
            'error_type': type(exc).__name__, 'reason': str(exc), 'config': config,
            'cuda_peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
        raise
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _datasets(root, config):
    from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
    from airwatch.data.ku_leuven_training_v2 import A800Dataset
    from training.a800_data import FROZEN_SPLIT_SHA256, read_csv
    root = Path(root).resolve()
    manifest_path = root/'manifests/known-split-v1/known-split-assignments.csv'
    if sha256_file(manifest_path) != FROZEN_SPLIT_SHA256:
        raise ValueError('frozen split assignment identity changed')
    rows = read_csv(manifest_path)
    assignments = {row['recording_id']: row for row in rows}
    if len(assignments) != len(rows):
        raise ValueError('duplicate recording assignments')
    train_class = KULeuvenMaterializedDataset if config['data']['train_artifact'] == 'v1' else A800Dataset
    train = train_class(bundle_path(root, config['data']['train_root']), split='train', verify_hashes=True)
    try:
        validation = KULeuvenMaterializedDataset(bundle_path(root, config['data']['validation_root']), split='validation', verify_hashes=True)
        validate_dataset_assignments(train, assignments,
            windows_per_recording=32 if config['data']['train_artifact'] == 'v1' else 256)
        validate_dataset_assignments(validation, assignments, windows_per_recording=32)
        for dataset in (train, validation):
            if dataset.metadata['source_split_assignments']['sha256'] != FROZEN_SPLIT_SHA256:
                raise ValueError('dataset provenance references another split assignment')
    except Exception:
        train.close()
        if 'validation' in locals():
            validation.close()
        raise
    return train, validation


def validate_dataset_assignments(dataset, assignments: dict, *, windows_per_recording: int) -> None:
    """Admit every row against the immutable split, independent of bootstrap."""
    if dataset.split not in {'train', 'validation'}:
        raise ValueError('training refuses sealed split assignments')
    expected = {rid for rid, row in assignments.items() if row['split'] == dataset.split}
    if set(dataset.recording_ids) != expected:
        raise ValueError('dataset recording membership differs from the frozen split')
    seen, counts = set(), Counter()
    for sample in dataset.samples:
        row = assignments.get(sample.recording_id)
        if row is None or any(str(getattr(sample, key)) != str(row[key])
                for key in ('archive_id', 'label_index', 'member_index', 'member_path', 'member_sha256', 'split')):
            raise ValueError('dataset sample provenance differs from the frozen assignment')
        if sample.window_id in seen:
            raise ValueError('duplicate dataset window ID')
        seen.add(sample.window_id)
        counts[sample.recording_id] += 1
    if set(counts) != expected or any(count != windows_per_recording for count in counts.values()):
        raise ValueError('recording window coverage differs from the fixed training protocol')


def run_matrix(root: Path, allocation: Allocation, *, stop_requested=None) -> int:
    allocation.recheck()
    root = Path(root).resolve()
    if allocation.root != root or allocation.phase != 'train':
        raise ValueError('matrix allocation belongs to another root or phase')
    configs = load_matrix(root)
    code = source_identity(root)
    # Resumable runs take priority; the remaining matrix retains A/B/C/D then seed order.
    def priority(config):
        path = root / 'outputs/runs' / config['run_id'] / 'run-status.json'
        if path.exists() and json.loads(path.read_text())['state'] in {'running', 'interrupted'}:
            return 0
        return 1
    for config in sorted(configs, key=priority):
        if allocation.should_stop() or (stop_requested and stop_requested()):
            return 75
        output = root / 'outputs/runs' / config['run_id']
        status_path = output / 'run-status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding='utf-8'))
            if status['state'] in {'completed', 'failed'}:
                if status['state'] == 'failed' and status.get('reason') != 'failed_no_eligible_checkpoint':
                    raise ValueError(f'unhandled failed run: {config["run_id"]}')
                checkpoint = load_checkpoint(output / 'last.pt')
                if (checkpoint['global_step'] != 9200 or checkpoint['identity']['code'] != code
                        or checkpoint['config'] != config
                        or status['checkpoint_sha256'] != sha256_file(output/'last.pt')):
                    raise ValueError('completed run has incomplete budget or changed source identity')
                train, validation = _datasets(root, config)
                try:
                    if checkpoint['identity']['data'] != {'train': _dataset_identity(train), 'validation': _dataset_identity(validation)}:
                        raise ValueError('completed run dataset identity changed')
                    if status['state'] == 'completed':
                        load_checkpoint(output/'best.pt', expected_identity=checkpoint['identity'])
                        if status['best_checkpoint_sha256'] != sha256_file(output/'best.pt'):
                            raise ValueError('completed run selected checkpoint identity changed')
                finally:
                    train.close()
                    validation.close()
                continue
        train, validation = _datasets(root, config)
        try:
            result = run_training(config, train, validation, output,
                device=f'cuda:{allocation.gpu_index}', allocation=allocation,
                code_identity=code, stop_requested=stop_requested)
            if result['state'] == 'interrupted':
                return 75
        finally:
            train.close()
            validation.close()
    return 0


class _BalancedSmokeDataset(Dataset):
    """One real anchor per known class, only for isolated engineering checks."""
    def __init__(self, dataset):
        indices = {}
        for index, sample in enumerate(dataset.samples):
            indices.setdefault(sample.label_index, index)
        if set(indices) != {0, 1, 2}:
            raise ValueError('resume smoke requires all three known classes')
        self.dataset = dataset
        self.indices = tuple(indices[label] for label in range(3))
        self.samples = tuple(dataset.samples[index] for index in self.indices)
        self.split = dataset.split
        self.data_identity = {'scope': 'balanced engineering subset',
                              'parent': _dataset_identity(dataset), 'indices': list(self.indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        values, label, _ = self.dataset[self.indices[index]]
        return values, label, index

    def sample_metadata(self, index):
        return self.samples[index]


def _smoke_validation(model, dataset, device, *, batch_size=64, allocation=None):
    """A bounded loss check; no member metrics or checkpoint-selection claim."""
    if allocation is not None and allocation.should_stop():
        raise InterruptedError('allocation ended during resume smoke')
    model.eval()
    with torch.inference_mode():
        items = [dataset[index] for index in range(len(dataset))]
        inputs = torch.stack([item[0] for item in items]).to(device)
        labels = torch.tensor([item[1] for item in items], device=device)
        loss = float(torch.nn.functional.cross_entropy(model(inputs).float(), labels).cpu())
    if not math.isfinite(loss):
        raise ValueError('nonfinite engineering validation loss')
    return {'rank': [-loss] * 4, 'loss': loss, 'scope': 'engineering loss only; no formal metrics'}


def _assert_recovery_equal(left, right, field):
    if isinstance(left, torch.Tensor):
        equal = isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        equal = isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, dict):
        equal = isinstance(right, dict) and set(left) == set(right)
        if equal:
            for key in left:
                _assert_recovery_equal(left[key], right[key], f'{field}.{key}')
    elif isinstance(left, (tuple, list)):
        equal = type(left) is type(right) and len(left) == len(right)
        if equal:
            for index, (first, second) in enumerate(zip(left, right)):
                _assert_recovery_equal(first, second, f'{field}[{index}]')
    else:
        equal = left == right
    if not equal:
        raise ValueError(f'exact resume equivalence failed at {field}')


def verify_resume_equivalence(config, train_dataset, validation_dataset, output_dir, *,
                              device='cpu', allocation=None, code_identity=None,
                              model_factory=None, stop_requested=None):
    """Four continuous updates versus two + serialized recovery + two updates.

    Real training, optimizer, scheduler, augmentation and RNG state are compared
    exactly; timing is deliberately excluded. Unsupported deterministic BF16
    operations fail this gate rather than introducing a numerical tolerance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir/'resume-equivalence.json'
    report = {'schema_version': '2.0', 'state': 'running', 'kind': 'engineering_smoke',
              'formal_result': False, 'exact_equal': False, 'continuous_steps': 4,
              'interruption_step': 2, 'anchors_per_batch': 3, 'device': str(device),
              'run_results': [],
              'precision': 'bf16' if torch.device(device).type == 'cuda' else 'fp32',
              'limitation': 'Bounded engineering equivalence; full-budget CUDA results remain unmeasured.'}
    atomic_json(report_path, report)
    try:
        config = copy.deepcopy(config)
        config.update(kind='engineering_smoke', run_id=f'engineering-smoke-resume-{config["model"]["name"]}')
        config['training'].update(anchors_per_batch=3, steps_per_epoch=4, logical_epochs=1,
                                  total_steps=4, warmup_steps=1, js_warmup_steps=1,
                                  precision=report['precision'])
        train, validation = _BalancedSmokeDataset(train_dataset), _BalancedSmokeDataset(validation_dataset)
        kwargs = dict(device=device, allocation=allocation, code_identity=code_identity,
                      model_factory=model_factory, evaluator=_smoke_validation, stop_requested=stop_requested)
        for directory, boundary in (('continuous', None), ('resumed', 2), ('resumed', None)):
            result = run_training(config, train, validation, output_dir/directory,
                                  stop_after_steps=boundary, **kwargs)
            report['run_results'].append({'directory': directory, 'stop_after_steps': boundary,
                                          'state': result['state'], 'global_step': result['global_step']})
            expected = 'interrupted' if boundary == 2 else 'completed'
            if result['state'] != expected or (boundary == 2 and result['global_step'] != 2):
                if result['state'] == 'interrupted':
                    raise InterruptedError('allocation or stop ended resume equivalence check')
                raise ValueError('resume equivalence training did not complete its engineering budget')
        continuous = load_checkpoint(output_dir/'continuous/last.pt')
        resumed = load_checkpoint(output_dir/'resumed/last.pt')
        fields = sorted((CHECKPOINT_FIELDS - {'timing'}) | {'best_training_state'})
        for field in fields:
            _assert_recovery_equal(continuous[field], resumed[field], field)
        report.update(state='completed', exact_equal=True, compared_fields=fields,
                      train_identity=train.data_identity, validation_identity=validation.data_identity,
                      checkpoints={name: sha256_file(output_dir/name/'last.pt') for name in ('continuous', 'resumed')})
        atomic_json(report_path, report)
        return report
    except Exception as exc:
        report.update(state='interrupted' if isinstance(exc, (InterruptedError, AllocationUnavailable)) else 'failed',
                      reason=f'{type(exc).__name__}: {exc}')
        atomic_json(report_path, report)
        raise


def run_gpu_preflight(root: Path, allocation: Allocation, *, stop_requested=None) -> dict:
    """Measure fixed-batch BF16 train steps and FP32 validation after allocation.

    These isolated engineering-smoke models are discarded. Measurements include
    three steps per architecture, one full validation and checkpoint write.
    Each architecture also passes a separate four-step serialized BF16 resume
    equivalence check. The estimate is never an observed full run.
    """
    if not isinstance(allocation, Allocation):
        raise AllocationUnavailable('CUDA preflight requires an explicit verified allocation')
    allocation.recheck()
    from training.a800_evaluation import evaluate_validation
    from training.a800_objective import make_paired_views, paired_objective
    root = Path(root).resolve()
    device = torch.device(f'cuda:{allocation.gpu_index}')
    report = {'schema_version': '2.0', 'kind': 'engineering_smoke', 'state': 'running',
              'gpu_uuid': allocation.gpu_uuid, 'protocol': PROTOCOL, 'code': source_identity(root),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'formal_result': False, 'models': []}
    report_path = root/'outputs/evidence/gpu-preflight.json'
    atomic_json(report_path, report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the allocated environment')
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unavailable; fixed protocol preflight failed')
        for experiment in ('A', 'D'):
            allocation.recheck()
            config = make_run_config(experiment, SEEDS[0])
            configure_determinism(config['seed'], config['training'], device)
            train, validation = _datasets(root, config)
            try:
                batch = torch.stack([train[index][0] for index in range(128)])
                labels = torch.tensor([train[index][1] for index in range(128)], device=device)
                window_ids = [train.sample_metadata(index).window_id for index in range(128)]
                counts = torch.bincount(torch.tensor([sample.label_index for sample in train.samples]), minlength=3)
                weights = counts.float().reciprocal()
                weights = (weights / weights.mean()).to(device)
                model = _default_model(config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                torch.cuda.reset_peak_memory_stats(device)
                seconds = []
                for step in range(3):
                    if allocation.should_stop() or (stop_requested and stop_requested()):
                        raise InterruptedError('allocated time ended during CUDA preflight')
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    first, second = make_paired_views(batch, window_ids, run_seed=SEEDS[0],
                        logical_epoch=0, draw_positions=list(range(step * 128, (step + 1) * 128)))
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss, _ = paired_objective(model, first.to(device), second.to(device), labels,
                            js_weight=.1 if experiment == 'D' else 0., class_weights=weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('CUDA preflight returned a non-finite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    seconds.append(time.perf_counter() - started)
                started = time.perf_counter()
                evaluate_validation(model, validation, device, batch_size=64, allocation=allocation)
                torch.cuda.synchronize(device)
                validation_seconds = time.perf_counter() - started
                started = time.perf_counter()
                snapshot = _cpu_copy({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
                with tempfile.TemporaryFile(dir=report_path.parent) as temporary:
                    torch.save(snapshot, temporary)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    serialized_bytes = temporary.tell()
                save_seconds = time.perf_counter() - started
                steady_step = sum(seconds[1:]) / len(seconds[1:])
                peak_memory = torch.cuda.max_memory_allocated(device)
                del optimizer, model, snapshot, first, second, batch
                torch.cuda.empty_cache()
                smoke_parent = root/'outputs/engineering-smoke'
                smoke_parent.mkdir(parents=True, exist_ok=True)
                session_dir = Path(tempfile.mkdtemp(prefix=f'resume-{config["model"]["name"]}-', dir=smoke_parent))
                resume_report = verify_resume_equivalence(config, train, validation, session_dir/'check',
                    device=device, allocation=allocation, code_identity=report['code'], stop_requested=stop_requested)
                report['models'].append({'model': config['model']['name'], 'training_steps': 3,
                    'data_identity': {'train': _dataset_identity(train), 'validation': _dataset_identity(validation)},
                    'anchors_per_batch': 128, 'inputs_per_forward': 256, 'step_seconds': seconds,
                    'validation_seconds': validation_seconds, 'save_seconds': save_seconds,
                    'serialized_bytes': serialized_bytes,
                    'peak_memory_allocated_bytes': peak_memory,
                    'resume_equivalence': dict(resume_report,
                        evidence_path=(session_dir/'check/resume-equivalence.json').relative_to(root).as_posix()),
                    'estimated_full_run_seconds': 9200 * steady_step + 40 * (validation_seconds + save_seconds),
                    'estimate_includes_validation_and_save': True,
                    'estimate_limitations': 'three-step measurement; storage/load contention and data-loader variation remain unmeasured'})
                atomic_json(report_path, report)
            finally:
                train.close()
                validation.close()
                torch.cuda.empty_cache()
        report['state'] = 'completed'
        atomic_json(report_path, report)
        return report
    except Exception as exc:
        report.update(state='interrupted' if isinstance(exc, (InterruptedError, AllocationUnavailable)) else 'failed',
                      reason=f'{type(exc).__name__}: {exc}')
        atomic_json(report_path, report)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=['train', 'finalize'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--session-minutes', type=int, required=True)
    parser.add_argument('--allocation-confirmed', action='store_true')
    args = parser.parse_args(argv)
    stop = {'requested': False}
    def handle_stop(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    try:
        allocation = require_allocation(args.root, gpu_index=args.gpu, session_minutes=args.session_minutes,
            allocation_confirmed=args.allocation_confirmed, phase=args.phase)
        allocation = replace(allocation, stop_requested=lambda: stop['requested'])
        # Verify the upload payload once per allocated invocation, before any
        # CUDA work. Dataset/source identities are rechecked by run boundaries.
        from training.build_a800_bundle import verify_integrity
        verify_integrity(args.root)
        from training.a800_environment import verify_environment
        verify_environment(args.root, require_linux=True)
        allocation.recheck()
        if args.phase == 'train':
            run_gpu_preflight(args.root, allocation, stop_requested=lambda: stop['requested'])
            return run_matrix(args.root, allocation, stop_requested=lambda: stop['requested'])
        from training.a800_evaluation import run_finalization
        result = run_finalization(args.root, allocation=allocation)
        return 75 if isinstance(result, dict) and result.get('state') == 'interrupted' else 0
    except (AllocationUnavailable, InterruptedError) as exc:
        print(json.dumps({'state': 'waiting_or_interrupted', 'reason': str(exc)}), flush=True)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
