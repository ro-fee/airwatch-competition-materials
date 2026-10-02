#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AIRWATCH_PYTHON:-/root/lyh_workspace/envs/airwatch-a800/bin/python}"
if [[ $# -ne 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/export_results.sh /absolute/path/results.tar.gz' >&2
  exit 2
fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=''
exec "$PYTHON" -m training.a800_delivery package --root "$ROOT" --output "$1"
