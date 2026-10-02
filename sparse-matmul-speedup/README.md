# Dense matmuls run faster on sparse weights (but only because of power)

Code and raw results for the post *Zeros make dense matmuls faster, but not the way you'd hope*.

We zero a fraction of an 8192x8192 bf16 weight, keep it stored densely, and time `x @ w` (cuBLAS, no sparse
kernel) replayed from a CUDA graph. We sweep the number of input rows `M` (1024 to 16384) and the amount of sparsity
(5% to 99%; random and magnitude pruning), with free-running SM clocks and with the clock locked at a range of values.

```bash
./run.sh <python-with-torch> 1000 1200 1400 1600 1800 1980   # unlocked, then one run per locked MHz (needs sudo)
uv run --with matplotlib python plot_results.py
```

Results live in `results/<gpu>/<tag>.json` (+ `.png`), one directory per GPU, with a `clock_sweep.png` per GPU:

| GPU | Runs |
|-----|------|
| H100 80GB HBM3 (700 W) | `results/h100/unlocked.json`, `locked-{1000,1200,1400,1600,1800,1980}mhz.json` |
| B200 (1000 W) | `results/b200/unlocked.json`, `locked-{1000,1200,1400,1600,1800,1965}mhz.json` |

Each timing is the median of 5 rounds; configs are interleaved and the order alternates each round. `sm_mhz` and
`watts` are NVML samples taken while the replays run, so treat `watts` as indicative (NVML power updates slower
than the shortest runs).
