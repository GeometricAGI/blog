#!/usr/bin/env bash
# Runs the sweep with free-running clocks, then with the SM clock locked.
# Usage: ./run.sh <python> [locked_mhz]    (locking needs sudo)
set -euo pipefail
PY=${1:-python}
LOCK=${2:-1200}
cd "$(dirname "$0")"
"$PY" sparse_matmul_bench.py --tag unlocked
trap 'sudo nvidia-smi -rgc >/dev/null' EXIT
sudo nvidia-smi -lgc "$LOCK,$LOCK"
"$PY" sparse_matmul_bench.py --tag "locked-${LOCK}mhz"
