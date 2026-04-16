# Reproducing the benchmark results and figures

## Setup

Requires Python >= 3.12 and an NVIDIA GPU with CUDA support.

```bash
cd torch-compile-mode-analysis
uv sync
```

## Running the benchmark

`torch_compile_mode_study.py` profiles 23 workloads across four domains (LLM, VLM, diffusion, RL/loss) under every `torch.compile` mode (`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`) plus an eager baseline. Each (workload, mode) combination runs in an isolated subprocess for clean CUDA state.

```bash
uv run python torch_compile_mode_study.py
```

Output (saved to `benchmark_results/torch_compile_mode_study/`):
- `raw_results.csv` -- one row per (shape preset, workload, mode, tag)
- `summary.json` -- aggregate statistics (wins, ranks, geometric-mean speedups)
- `report.md` -- human-readable Markdown summary
- `plots/*.png` -- per-workload bar charts and a geometric-mean speedup chart

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `benchmark_results/torch_compile_mode_study` | Directory for outputs |
| `--dtype` | `bfloat16` | Data type (`float16` or `bfloat16`) |
| `--warmup` | `100` | Warm-up time budget (ms) for Triton `do_bench` |
| `--rep` | `1000` | Repetition time budget (ms) for Triton `do_bench` |
| `--timeout-s` | `1200` | Per-subprocess timeout in seconds |
| `--shape-presets` | `decode,prefill` | Comma-separated shape presets to benchmark |
| `--analyze-only` | off | Recompute summary/report/plots from an existing CSV |

## Re-analyzing existing results

To regenerate the summary, report, and plots without re-running the benchmarks:

```bash
uv run python torch_compile_mode_study.py --analyze-only
```
