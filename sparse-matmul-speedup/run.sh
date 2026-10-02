#!/usr/bin/env bash
# Runs the sweep with free-running clocks, then once per locked SM clock.
# Usage: [BENCH_ARGS='--shapes ... --suite x'] ./run.sh <python> [<mhz> ...]    (locking needs sudo)
set -euo pipefail
PY=${1:-python}
shift || true
cd "$(dirname "$0")"
"$PY" sparse_matmul_bench.py --tag unlocked ${BENCH_ARGS:-}
trap 'sudo nvidia-smi -rgc >/dev/null' EXIT
for mhz in "$@"; do
    sudo nvidia-smi -lgc "$mhz,$mhz" >/dev/null
    "$PY" sparse_matmul_bench.py --tag "locked-${mhz}mhz" ${BENCH_ARGS:-}
done
