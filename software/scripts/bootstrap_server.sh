#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX='/root/lyh_workspace/envs/airwatch-a800'
cd "$ROOT"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { printf '%s\n' 'Ubuntu/Linux x86_64 required'; exit 2; }
command -v conda >/dev/null || { printf '%s\n' 'Initialize Conda in this shell first'; exit 2; }
sha256sum -c SHA256SUMS
# This stdlib verifier works even with the server base Python; it does not import torch.
conda run -n base python -m training.build_a800_bundle verify --root "$ROOT"
if [[ ! -e "$PREFIX" ]]; then
  conda create -y -p "$PREFIX" python=3.10.21 pip
  sha256sum environment/requirements-linux-lock.txt | cut -d ' ' -f1 > "$PREFIX/.airwatch-a800-environment.sha256"
else
  [[ -f "$PREFIX/.airwatch-a800-environment.sha256" ]] || { printf '%s\n' 'Existing prefix has no AirWatch ownership record; leaving it unchanged'; exit 2; }
  EXPECTED="$(sha256sum environment/requirements-linux-lock.txt | cut -d ' ' -f1)"
  [[ "$(cat "$PREFIX/.airwatch-a800-environment.sha256")" == "$EXPECTED" ]] || { printf '%s\n' 'Existing prefix belongs to a different environment specification'; exit 2; }
fi
PYTHON="$PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { printf '%s\n' 'Existing prefix is not a Python environment'; exit 2; }
"$PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,10,21), "Existing prefix has wrong Python; do not delete or overwrite it"'
if [[ -f "$PREFIX/.airwatch-a800-installed" ]]; then
  "$PYTHON" -m training.a800_environment --root "$ROOT" --require-linux
fi
"$PYTHON" -m pip install --require-hashes --no-deps -r environment/requirements-linux-lock.txt
"$PYTHON" -m pip check
"$PYTHON" -m training.a800_environment --root "$ROOT" --require-linux
touch "$PREFIX/.airwatch-a800-installed"
mkdir -p outputs/evidence/environment
"$PYTHON" -m pip freeze --all > outputs/evidence/environment/pip-freeze.txt
AIRWATCH_PYTHON="$PYTHON" bash scripts/run_cpu_checks.sh
printf '%s\n' 'CPU bootstrap complete. GPU work still requires an actual allocation.'
