"""A800 measured GPU benchmarks and portable, reloadable runtime releases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_preprocessing import PREPROCESSING_ID, preprocess_iq_window
from airwatch.inference.uav_a800_contract import CODE_FILES, load_contract, member_scores, predict_member
from training.a800_evaluation import atomic_json, check_allocation, start_finalization, read_json, verify_freeze, _load_model, cached_shard, _digest, relative_path


def environment():
    return dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__), numpy=np.__version__,
                cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())


def write_runtime_contract(model, architecture, prototypes, thresholds, output_dir, *, source_root, model_version):
    """Package supplied frozen values. No metrics, thresholds or models are fitted."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing release: {output}')
    output.mkdir(parents=True)
    source_root = Path(source_root).resolve()
    preprocessing = dict(sample_rate_hz=100000000, windows_per_member=32, window_samples=4096,
                         channels=['in_phase', 'quadrature'], preprocessing_id=PREPROCESSING_ID)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / 'model.pt')
    atomic_json(output / 'label-map.json', {'frysky': 0, 'spektrum_dx4e': 1, 'dji_mini2_rc': 2})
    atomic_json(output / 'preprocessing.json', preprocessing)
    atomic_json(output / 'open-set.json', dict(primary_score='cosine', comparison='score>=threshold', direction='higher_is_known',
                                             scores=thresholds, prototypes=np.asarray(prototypes).tolist()))
    atomic_json(output / 'environment.json', environment())
    hashes = {}
    for relative in CODE_FILES:
        source = source_root / relative
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = sha256_file(source)
    artifacts = {name: dict(sha256=sha256_file(output / name), size_bytes=(output / name).stat().st_size)
                 for name in ('model.pt', 'label-map.json', 'preprocessing.json', 'open-set.json', 'environment.json')}
    contract = dict(schema_version='2.0', artifact_type='airwatch_a800_runtime_v2', model_version=model_version,
                    architecture=architecture, precision='fp32', input=preprocessing, primary_score='cosine',
                    direction='higher_is_known', artifacts=artifacts, code_hashes=hashes,
                    parameters=sum(p.numel() for p in model.parameters()),
                    source_loading='Put release/source first on PYTHONPATH; imported code must match shipped hashes.',
                    limitations=['Three known RF control sources, not generic UAV presence detection.',
                                 'A800 benchmark does not establish performance on an unmeasured target GPU.'])
    path = output / 'runtime-contract.json'
    atomic_json(path, contract)
    (output / 'runtime-contract.sha256').write_text(sha256_file(path) + '\n', encoding='ascii')
    return path


def _selected(payload):
    selected = payload['selection']['selected_experiment']
    rows = [run for run in payload['runs'] if run['config']['experiment'] == selected and run['config']['seed'] == 20260909]
    if len(rows) != 1:
        raise ValueError('Missing fixed deployment seed for selected experiment')
    return rows[0]


def _validation_input(root, config):
    from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
    with KULeuvenMaterializedDataset(root / config['data']['validation_root'], split='validation') as data:
        recording = data.samples[0].recording_id
        indexes = [i for i, sample in enumerate(data.samples) if sample.recording_id == recording]
        if len(indexes) != 32:
            raise ValueError('Benchmark input requires one complete validation member')
        return np.stack([data[i][0].numpy() for i in indexes]), recording


def _rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        if sys.platform == 'linux':
            with open('/proc/self/statm', encoding='ascii') as handle:
                return int(handle.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        return None


def benchmark_frozen(freeze, *, allocation=None):
    start_finalization(allocation)
    payload = verify_freeze(freeze)
    root, run = Path(payload['_root']), _selected(payload)
    if root != allocation.root.resolve():
        raise ValueError('Benchmark and allocation refer to different roots')
    output = root / 'outputs/evidence/benchmark'
    output.mkdir(parents=True, exist_ok=True)
    freeze_identity = sha256_file(freeze)
    measured_environment = environment()
    identity = _digest(dict(freeze_sha256=freeze_identity, gpu_uuid=allocation.gpu_uuid, environment=measured_environment))
    final = output / 'benchmark.json'
    if final.exists():
        report = read_json(final)
        if report['freeze_sha256'] != freeze_identity or report.get('benchmark_identity') != identity:
            raise ValueError('Benchmark freeze, GPU or environment changed')
        for item in report['raw_files']:
            if sha256_file(output / item['path']) != item['sha256']:
                raise ValueError('Benchmark raw timing file changed')
        return final
    config = run['config']
    model, _ = _load_model(root, config, root / run['checkpoint'], 'cuda', allocation=allocation)
    calibration = read_json(root / run['calibration'])['result']
    windows, recording_id = _validation_input(root, config)
    prototypes = np.asarray(calibration['prototypes'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries, raw_files = [], []
    for count in (1, 32):
        values = windows[:count]
        check_allocation(allocation)
        tensor = torch.from_numpy(values.copy()).cuda().float()
        for scope in ('model', 'end_to_end'):
            def operation():
                with torch.inference_mode():
                    batch = tensor if scope == 'model' else torch.from_numpy(np.stack([preprocess_iq_window(w) for w in values])).cuda().float()
                    if scope == 'model':
                        return model(batch)
                    features = model.forward_features(batch)
                    logits = model.classifier(features)
                    feature, logits = features.cpu().numpy(), logits.cpu().numpy()
                    if count == 32:
                        return member_scores(logits, feature, prototypes)
                    # Single-window benchmark includes transfer/softmax, without
                    # presenting a one-window result as a valid member decision.
                    return torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            for _ in range(100):
                check_allocation(allocation)
                operation()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings, memories = [], []
            for chunk in range(20):
                check_allocation(allocation)
                def compute():
                    samples = []
                    for _ in range(50):
                        check_allocation(allocation)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        operation()
                        torch.cuda.synchronize()
                        samples.append(time.perf_counter() - start)
                    return dict(seconds=samples, cuda_peak_bytes=int(torch.cuda.max_memory_allocated()), process_rss_bytes=_rss_bytes())
                path = output / f'{count}-{scope}-{chunk:02d}.json'
                measured = cached_shard(path, identity, compute)
                timings.extend(measured['seconds'])
                memories.append(measured['cuda_peak_bytes'])
                raw_files.append(dict(path=path.name, sha256=sha256_file(path)))
            if len(timings) != 1000 or not np.isfinite(timings).all() or min(timings) <= 0:
                raise ValueError('Benchmark did not produce 1000 valid actual measurements')
            summaries.append(dict(windows=count, scope=scope, warmup_per_process=100, measurements=1000,
                                  p50_seconds=float(np.quantile(timings, .5)), p95_seconds=float(np.quantile(timings, .95)),
                                  windows_per_second=count / float(np.mean(timings)), cuda_peak_bytes=max(memories)))
    report = dict(freeze_sha256=freeze_identity, benchmark_identity=identity, gpu_uuid=allocation.gpu_uuid,
                  run_id=config['run_id'], validation_member=recording_id,
                  validation_input_sha256=__import__('hashlib').sha256(windows.tobytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), precision='fp32', summaries=summaries, raw_files=raw_files,
                  parameters=sum(p.numel() for p in model.parameters()), checkpoint_bytes=(root / run['checkpoint']).stat().st_size,
                  runtime_state_dict_tensor_bytes=sum(t.numel() * t.element_size() for t in model.state_dict().values()),
                  process_rss_bytes=_rss_bytes(), environment=measured_environment, target_computer_status='not_measured')
    atomic_json(final, report)
    del model
    torch.cuda.empty_cache()
    return final


def export_release(freeze, results, output_dir, *, allocation=None):
    start_finalization(allocation)
    payload = verify_freeze(freeze)
    root, output = Path(payload['_root']), Path(output_dir).resolve()
    if root != allocation.root.resolve():
        raise ValueError('Export and allocation refer to different roots')
    completion = read_json(Path(results).parent / 'evaluation-completed.json')
    if completion.get('freeze_sha256') != sha256_file(freeze) or sha256_file(results) != next((a['sha256'] for a in completion['artifacts'] if a['path'] == 'metrics.json'), None):
        raise ValueError('Export requires completed immutable test evidence')
    if output.exists():
        status = read_json(output / 'release-completed.json')
        if status['freeze_sha256'] != sha256_file(freeze):
            raise ValueError('Existing release belongs to another freeze')
        for item in status['files']:
            if sha256_file(output / item['path']) != item['sha256']:
                raise ValueError('Existing release file changed')
        load_contract(output / 'runtime-contract.json', device='cpu')
        return output
    run = _selected(payload)
    config = run['config']
    model, _ = _load_model(root, config, root / run['checkpoint'], 'cuda', allocation=allocation)
    calibration = read_json(root / run['calibration'])['result']
    staging = output.with_name(output.name + '.staging')
    if staging.exists():
        # A interrupted staging payload is preserved for audit; a fresh sibling
        # avoids silently trusting or deleting unfinished export files.
        staging = output.with_name(output.name + f'.staging-{time.time_ns()}')
    contract_path = write_runtime_contract(model, config['model']['name'], calibration['prototypes'], calibration['scores'],
                                           staging, source_root=root, model_version=sha256_file(freeze))
    check_allocation(allocation)
    contract = load_contract(contract_path, device='cuda')
    values, recording_id = _validation_input(root, config)
    check_allocation(allocation)
    predicted = predict_member(contract, values, sample_rate_hz=100000000)
    check_allocation(allocation)
    with torch.inference_mode():
        prepared = torch.from_numpy(np.stack([preprocess_iq_window(w) for w in values])).cuda()
        features = model.forward_features(prepared)
        logits = model.classifier(features).cpu().numpy()
        expected = member_scores(logits, features.cpu().numpy(), np.asarray(calibration['prototypes']))
    errors = dict(logits_max_absolute=float(np.max(np.abs(np.asarray(predicted['logits']) - logits))),
                  scores_max_absolute=max(abs(predicted['scores'][s] - expected[s]) for s in ('cosine', 'msp', 'energy')))
    expected_decision = 'known' if expected['cosine'] >= calibration['scores']['cosine']['threshold'] else 'unknown'
    if predicted['label_index'] != expected['prediction'] or predicted['decision'] != expected_decision or errors['logits_max_absolute'] > 1e-5 or errors['scores_max_absolute'] > 1e-6:
        raise ValueError(f'Export reload comparison failed: {errors}')
    atomic_json(staging / 'reload-verification.json', dict(validation_member=recording_id, actual_errors=errors,
                                                        class_equal=True, decision_equal=True, precision='fp32'))
    shutil.copy2(freeze, staging / 'freeze.json')
    shutil.copy2(results, staging / 'metrics.json')
    benchmark = root / 'outputs/evidence/benchmark/benchmark.json'
    if not benchmark.is_file() or read_json(benchmark).get('freeze_sha256') != sha256_file(freeze):
        raise ValueError('Export requires completed frozen A800 benchmark')
    shutil.copy2(benchmark, staging / 'benchmark.json')
    atomic_json(staging / 'metrics-reference.json', dict(metrics_sha256=sha256_file(results), freeze_sha256=sha256_file(freeze), benchmark_sha256=sha256_file(benchmark)))
    files = [dict(path=p.relative_to(staging).as_posix(), sha256=sha256_file(p)) for p in sorted(staging.rglob('*')) if p.is_file()]
    atomic_json(staging / 'release-completed.json', dict(freeze_sha256=sha256_file(freeze), files=files, status='completed'))
    lines = [f'{sha256_file(p)}  {p.relative_to(staging).as_posix()}' for p in sorted(staging.rglob('*')) if p.is_file()]
    (staging / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(staging, output)
    return output


def package_results(root, output=None):
    """Verify and archive existing completed results; never starts model inference."""
    import tarfile
    root = Path(root).resolve()
    output = Path(output or root / 'outputs/a800-results.tar.gz').resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite results archive: {output}')
    freeze = root / 'outputs/evidence/freeze.json'
    verify_freeze(freeze, root=root)
    for directory, marker, expected_key in (
        (root / 'outputs/release', 'release-completed.json', 'files'),
        (root / 'outputs/evidence/test', 'evaluation-completed.json', 'artifacts'),
    ):
        status = read_json(directory / marker)
        if status.get('status') != 'completed' or not status.get(expected_key) or status.get('freeze_sha256') != sha256_file(freeze):
            raise ValueError('Result package contains a mismatched freeze')
        from training.a800_evaluation import relative_path
        for item in status[expected_key]:
            if sha256_file(relative_path(directory, item['path'])) != item['sha256']:
                raise ValueError(f'Packaging checksum mismatch: {item["path"]}')
        if marker == 'release-completed.json':
            listed = sorted(status['files'] + [dict(path=marker, sha256=sha256_file(directory / marker))], key=lambda item: item['path'])
            expected_sums = {f'{item["sha256"]}  {item["path"]}' for item in listed}
            if set((directory / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()) != expected_sums:
                raise ValueError('Release SHA256SUMS does not match completed payload')
    benchmark_dir = root / 'outputs/evidence/benchmark'
    benchmark = read_json(benchmark_dir / 'benchmark.json')
    if benchmark.get('freeze_sha256') != sha256_file(freeze):
        raise ValueError('Result package is missing the completed GPU benchmark')
    for item in benchmark['raw_files']:
        if sha256_file(relative_path(benchmark_dir, item['path'])) != item['sha256']:
            raise ValueError('Raw GPU timing data changed')
    directories = [root / value for value in ('outputs/release', 'outputs/evidence', 'outputs/runs', 'outputs/progress', 'environment', 'training/configs/a800_v2')]
    files = sorted({p for folder in directories if folder.is_dir() for p in folder.rglob('*')
                    if p.is_file() and not p.name.endswith(('.tmp', '.pyc')) and '__pycache__' not in p.parts})
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError('Result package cannot follow external file links')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    os.replace(temporary, output)
    atomic_json(output.with_name(output.name + '.sha256.json'), dict(path=output.name, sha256=sha256_file(output), size_bytes=output.stat().st_size, files=len(files)))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('benchmark', 'export', 'package'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--session-minutes', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(argv)
    if args.phase == 'package':
        print(json.dumps({'output': str(package_results(args.root, args.output))}))
        return 0
    if args.freeze is None:
        parser.error('--freeze is required for benchmark/export')
    from training.a800_common import require_allocation, AllocationUnavailable
    try:
        payload = verify_freeze(args.freeze)
        root = Path(payload['_root'])
        allocation = require_allocation(root, gpu_index=args.gpu, session_minutes=args.session_minutes,
                                        allocation_confirmed=args.allocation_confirmed, phase='finalize')
        output = benchmark_frozen(args.freeze, allocation=allocation) if args.phase == 'benchmark' else export_release(args.freeze, root / 'outputs/evidence/test/metrics.json', root / 'outputs/release', allocation=allocation)
        print(json.dumps({'output': str(output)}))
        return 0
    except (InterruptedError, AllocationUnavailable) as exc:
        print(json.dumps({'status': 'interrupted', 'reason': str(exc)}))
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
