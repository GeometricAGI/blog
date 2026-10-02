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

Each graph rotates through `--copies` (default 8) independent weights, activations and outputs so consecutive uses of a tensor are far apart in L2 terms. Each timing is the median of 5 rounds; configs are interleaved and the order alternates each round. `sm_mhz` and
`watts` are NVML samples taken while the replays run, so treat `watts` as indicative (NVML power updates slower
than the shortest runs).

## DeepSeek-V4-Flash shapes

`results/<gpu>/dsv4-flash/{unlocked,locked-<mhz>mhz}.json` (+ `.png` from `plot_dsv4.py`) use the weight shapes of
DeepSeek-V4-Flash (`q_b_proj` 1024->32768, routed-expert `gate_up` 4096->4096 with rows scaled by 6/256, `lm_head`
4096->129280) at token counts 1 to 8192:

```bash
BENCH_ARGS="--suite dsv4-flash --shapes q_b_proj:1024:32768 expert_gate_up:4096:4096:0.0234375 lm_head:4096:129280 \
  --ms 1 8 32 128 512 2048 8192 --sparsities 0.25 0.5 0.7 0.9 0.99 --modes random" ./run.sh <python> 1200
uv run --with matplotlib python plot_dsv4.py
```

Weights are stored `(out, in)` and applied as `x @ w.t()`. The top-level `results/<gpu>/*.json` files (square 8192x8192)
were produced with the earlier `(in, out)` layout of commit `3cd0fac`.
