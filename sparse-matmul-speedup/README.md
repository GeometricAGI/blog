# Dense matmuls run faster on sparse weights (but only because of power)

Code and raw results for the post *Zeros make dense matmuls faster, but not the way you'd hope*.

We zero a fraction of an 8192x8192 bf16 weight, keep it stored densely, and time `x @ w` (cuBLAS, no sparse
kernel) replayed from a CUDA graph. We sweep the number of input rows `M` (1024 to 16384), the amount of sparsity
(random and magnitude pruning, plus 2:4), and run with free-running and locked SM clocks.

```bash
./run.sh <python-with-torch> [locked_mhz]   # runs unlocked, then locked (needs sudo for nvidia-smi -lgc)
uv run --with matplotlib python plot_results.py
```

Results live in `results/<gpu>/<tag>.json` (+ `.png`), one directory per GPU:

| GPU | Unlocked | Locked |
|-----|----------|--------|
| H100 80GB HBM3 (700 W) | `results/h100/unlocked.json` | `results/h100/locked-1200mhz.json` |
| B200 (1000 W) | `results/b200/unlocked.json` | `results/b200/locked-1000mhz.json` |

Each timing is the median of 5 rounds; configs are interleaved and the order alternates each round. `sm_mhz` and
`watts` are NVML samples taken while the replays run, so treat `watts` as indicative (NVML power updates slower
than the shortest runs).
