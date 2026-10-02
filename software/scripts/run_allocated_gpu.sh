#!/usr/bin/env bash
set -euo pipefail

# Actual allocation duration and confirmation are mandatory arguments. No polling
# for free GPUs and no process termination is performed by either gate.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AIRWATCH_PYTHON:-/root/lyh_workspace/envs/airwatch-a800/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' "Dedicated AirWatch Python is unavailable: $PYTHON" >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export AIRWATCH_ALLOCATION_STARTED_AT="$(date +%s)"
cd "$ROOT"

# The shell entry verifies allocation first. The dispatched Python entry checks
# it again and uses the same start timestamp, so rechecks never renew the lease.
"$PYTHON" -m training.a800_common --root "$ROOT" "$@"
exec "$PYTHON" -m training.a800_runner --root "$ROOT" "$@"
