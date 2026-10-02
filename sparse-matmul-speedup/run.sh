#!/usr/bin/env bash
# Runs the sweep with free-running clocks, then once per locked SM clock.
# Usage: ./run.sh <python> <mhz> [<mhz> ...]    (locking needs sudo)
set -euo pipefail
PY=${1:-python}
shift
cd "$(dirname "$0")"
"$PY" sparse_matmul_bench.py --tag unlocked
trap 'sudo nvidia-smi -rgc >/dev/null' EXIT
for mhz in "$@"; do
    sudo nvidia-smi -lgc "$mhz,$mhz" >/dev/null
    "$PY" sparse_matmul_bench.py --tag "locked-${mhz}mhz"
done
