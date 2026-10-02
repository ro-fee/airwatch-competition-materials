#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AIRWATCH_PYTHON:-/root/lyh_workspace/envs/airwatch-a800/bin/python}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
"$PYTHON" -m training.build_a800_bundle verify --root "$ROOT"
"$PYTHON" -m training.a800_cpu_checks --root "$ROOT" --require-linux
