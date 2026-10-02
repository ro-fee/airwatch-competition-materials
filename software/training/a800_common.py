"""Portable evidence writes and explicit, finite shared-GPU allocations.

This gate verifies an operator's allocation; it never schedules, polls for a free
GPU, sends signals to other processes, or infers permission from low utilization.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import tempfile
import time
from typing import Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    paths = sorted({path for directory in ('airwatch', 'training')
                    for path in (root / directory).rglob('*.py') if '__pycache__' not in path.parts}
                   | set((root/'training/configs').rglob('*.json')))
    if not paths:
        raise ValueError('source identity requires the bundled airwatch and training sources')
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def bundle_path(root: Path, relative: str) -> Path:
    """Resolve portable payload paths and reject drive, UNC and symlink escapes."""
    from pathlib import PureWindowsPath
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('expected a portable bundle relative path')
    if Path(relative).is_absolute() or PureWindowsPath(relative).drive or '..' in Path(relative).parts:
        raise ValueError('bundle path escape')
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('bundle path escape')
    return resolved


class AllocationUnavailable(RuntimeError):
    exit_code = 75


def disk_reserve_exhausted(root: Path) -> bool:
    """Stop new work with enough free space left to commit recovery state."""
    return shutil.disk_usage(root).free < 20 * 1024**3


def _gpu_snapshot(gpu_index: int, expected_uuid: str | None = None) -> dict:
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'],
            text=True, timeout=20, stderr=subprocess.STDOUT)
        rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
        gpu = next(row for row in rows if len(row) == 4 and int(row[0]) == gpu_index)
        uuid, name, memory = gpu[1].strip(), gpu[2].strip(), int(gpu[3])
        if not uuid.startswith('GPU-') or (expected_uuid and uuid != expected_uuid):
            raise AllocationUnavailable('allocated GPU UUID changed')
        if 'A800' not in name or memory < 79000:
            raise AllocationUnavailable('formal protocol requires the allocated A800 80GB')
        apps_text = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
             '--format=csv,noheader,nounits'], text=True, timeout=20, stderr=subprocess.STDOUT)
        apps = []
        for row in csv.reader(io.StringIO(apps_text), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 4:
                raise AllocationUnavailable('cannot verify GPU process inventory')
            if row[0].strip() == uuid:
                pid = int(row[1])
                if pid != os.getpid():
                    apps.append({'pid': pid, 'process': row[2].strip(), 'memory_mib': row[3].strip()})
        if apps:
            raise AllocationUnavailable(f'allocated GPU is occupied by other compute processes: {apps}')
        return {'gpu_uuid': uuid, 'name': name, 'memory_mib': memory,
                'occupancy_verification': 'nvidia_smi_visible_processes_only',
                'container_visibility_caveat': 'explicit operator/scheduler allocation remains required'}
    except AllocationUnavailable:
        raise
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError) as exc:
        raise AllocationUnavailable(f'cannot verify allocated GPU: {exc}') from exc


@dataclass(frozen=True)
class Allocation:
    root: Path
    gpu_index: int
    gpu_uuid: str
    phase: str
    session_minutes: int
    started_at: float
    allocation_source: str
    stop_requested: Callable[[], bool] | None = field(default=None, compare=False, repr=False)

    def remaining_seconds(self) -> float:
        return max(0.0, self.started_at + self.session_minutes * 60 - time.time())

    def should_stop(self, reserve_seconds: float = 300) -> bool:
        return ((self.stop_requested is not None and self.stop_requested())
                or self.remaining_seconds() <= reserve_seconds
                or disk_reserve_exhausted(self.root))

    def recheck(self) -> 'Allocation':
        if disk_reserve_exhausted(self.root):
            raise AllocationUnavailable('disk free space is below the 20 GiB checkpoint reserve')
        if self.should_stop():
            raise AllocationUnavailable('allocation is within the five-minute save/cleanup reserve')
        _gpu_snapshot(self.gpu_index, expected_uuid=self.gpu_uuid)
        return self


def require_allocation(root: Path, *, gpu_index: int, session_minutes: int,
                       allocation_confirmed: bool, phase: str) -> Allocation:
    root = Path(root).resolve()
    status_path = root / 'outputs/progress/allocation-status.json'
    try:
        if allocation_confirmed is not True:
            raise AllocationUnavailable('explicit allocation confirmation is required')
        if type(session_minutes) is not int or session_minutes <= 5:
            raise AllocationUnavailable('actual session_minutes must be an integer greater than the five-minute reserve')
        if type(gpu_index) is not int or gpu_index < 0 or phase not in {'train', 'finalize'}:
            raise AllocationUnavailable('valid GPU index and train/finalize phase required')
        started_at = float(os.environ.get('AIRWATCH_ALLOCATION_STARTED_AT', time.time()))
        now = time.time()
        if not (0 < started_at <= now + 1) or now - started_at >= session_minutes * 60 - 300:
            raise AllocationUnavailable('allocation start time is invalid or already within cleanup reserve')
        if disk_reserve_exhausted(root):
            raise AllocationUnavailable('disk free space is below the 20 GiB checkpoint reserve')
        snapshot = _gpu_snapshot(gpu_index)
        source = os.environ.get('SLURM_JOB_ID') or os.environ.get('AIRWATCH_ALLOCATION_SOURCE') or 'explicit_operator_confirmation'
        allocation = Allocation(root, gpu_index, snapshot['gpu_uuid'], phase, session_minutes, started_at, source)
        atomic_json(status_path, {'state': 'allocated', 'phase': phase, 'gpu_index': gpu_index,
            **snapshot, 'allocation_source': source, 'session_minutes': session_minutes,
            'started_at_unix': started_at, 'deadline_unix': started_at + session_minutes * 60,
            'pid': os.getpid(), 'hostname': socket.gethostname(),
            'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'automatic_scheduling': False})
        return allocation
    except (AllocationUnavailable, ValueError) as exc:
        atomic_json(status_path, {'state': 'waiting_for_allocation', 'phase': phase,
            'reason': str(exc), 'exit_code': 75, 'automatic_scheduling': False})
        if isinstance(exc, AllocationUnavailable):
            raise
        raise AllocationUnavailable(str(exc)) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=['train', 'finalize'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--session-minutes', type=int, required=True)
    parser.add_argument('--allocation-confirmed', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = require_allocation(args.root, gpu_index=args.gpu, session_minutes=args.session_minutes,
                                    allocation_confirmed=args.allocation_confirmed, phase=args.phase)
        print(json.dumps({'state': 'allocated', 'gpu_uuid': result.gpu_uuid,
                          'remaining_seconds': result.remaining_seconds()}))
        return 0
    except AllocationUnavailable as exc:
        print(json.dumps({'state': 'waiting_for_allocation', 'reason': str(exc)}))
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
