"""Does a dense matmul get faster when one operand is sparse (zeros stored densely)?

Times dense ``torch.matmul`` replayed through a CUDA graph with the weight zeroed to
different degrees. Also logs SM clock and power so a speedup can be attributed to
data-dependent power/clock effects rather than skipped work.
"""

import argparse
import json
import os
import re
import statistics

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover - NVML is optional
    NVML_HANDLE = None


def gpu_slug(name: str) -> str:
    """Turn a CUDA device name into a short directory name.

    Args:
        name: Device name, e.g. ``NVIDIA H100 80GB HBM3``.

    Returns:
        ``h100`` for ``NVIDIA H100 80GB HBM3``; falls back to a sanitized name.
    """
    m = re.search(r"\b([A-Z]\d{2,3}[A-Z]?)\b", name)
    return (m.group(1) if m else re.sub(r"\W+", "-", name)).lower()


def make_weight(k: int, n: int, sparsity: float, mode: str, dtype: torch.dtype) -> torch.Tensor:
    """Build a ``k x n`` weight with a given fraction of entries zeroed.

    Args:
        k: Rows of the weight (the contraction dimension).
        n: Columns of the weight.
        sparsity: Fraction of entries set to zero, in [0, 1).
        mode: ``random`` zeroes entries uniformly at random, ``magnitude`` zeroes the
            smallest-magnitude entries.
        dtype: Output dtype.

    Returns:
        A dense ``(k, n)`` CUDA tensor containing the zeros.
    """
    w = torch.randn(k, n, device="cuda", dtype=torch.float32)
    if sparsity > 0:
        if mode == "random":
            w = w * (torch.rand_like(w) >= sparsity)
        else:
            thresh = w.abs().flatten().kthvalue(int(sparsity * w.numel())).values
            w = w * (w.abs() > thresh)
    return w.to(dtype).contiguous()


def capture(x: torch.Tensor, w: torch.Tensor, reps: int):
    """Capture ``reps`` back-to-back matmuls ``x @ w`` into one CUDA graph.

    Args:
        x: Activation of shape ``(m, k)``.
        w: Weight of shape ``(k, n)``.
        reps: Matmuls per graph, to amortize the graph launch.

    Returns:
        The captured ``torch.cuda.CUDAGraph``.
    """
    out = torch.empty(x.shape[0], w.shape[1], device="cuda", dtype=x.dtype)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            torch.matmul(x, w, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            torch.matmul(x, w, out=out)
    return graph


def time_graph(graph, replays: int, reps: int) -> dict:
    """Time graph replays and sample clock/power at the end of the run.

    Args:
        graph: A captured CUDA graph.
        replays: Number of ``graph.replay()`` calls timed back to back.
        reps: Matmuls inside the graph (to convert to per-matmul time).

    Returns:
        Dict with ``us`` (microseconds per matmul), ``sm_mhz`` and ``watts``.
    """
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    sm_mhz = watts = None
    if NVML_HANDLE is not None:
        # sample while the queued replays are still executing
        sm_mhz = pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE, pynvml.NVML_CLOCK_SM)
        watts = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000
    torch.cuda.synchronize()
    return {"us": start.elapsed_time(end) * 1000 / (replays * reps), "sm_mhz": sm_mhz, "watts": watts}


def main() -> None:
    """Run the sweep and print a JSON results blob plus a summary table."""
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--ms", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--sparsities", type=float, nargs="+", default=[0, 0.05, 0.10, 0.25, 0.50, 0.70, 0.90, 0.99])
    p.add_argument("--modes", nargs="+", default=["random", "magnitude"])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tag", default="unlocked", help="Run label, e.g. 'unlocked' or 'locked-1200mhz'.")
    p.add_argument("--results-dir", default="results", help="Results go to <results-dir>/<gpu>/<tag>.json.")
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    # configs: (mode, sparsity), with one dense baseline.
    configs = [("dense", 0.0)] + [(m, s) for m in args.modes for s in args.sparsities if s > 0]
    weights = {}
    for mode, s in configs:
        weights[(mode, s)] = make_weight(args.k, args.n, s, "random" if mode == "dense" else mode, dtype)
        z = (weights[(mode, s)] == 0).float().mean().item()
        print(f"built {mode} {s}: actual zero frac {z:.3f}", flush=True)

    results = []
    for m in args.ms:
        x = torch.randn(m, args.k, device="cuda", dtype=dtype)
        # keep the whole sweep ~constant wall time: fewer reps for big problems
        flops = 2 * m * args.k * args.n
        reps = max(1, min(50, int(2e12 / flops)))
        replays = max(5, min(200, int(4e13 / (flops * reps))))
        graphs = {c: capture(x, weights[c], reps) for c in configs}
        samples = {c: [] for c in configs}
        for r in range(args.rounds):
            # alternate order each round so thermal/clock drift hits every config equally
            order = configs if r % 2 == 0 else configs[::-1]
            for c in order:
                samples[c].append(time_graph(graphs[c], replays, reps))
        for c in configs:
            us = [s["us"] for s in samples[c]]
            row = {
                "m": m, "mode": c[0], "sparsity": c[1], "us_median": statistics.median(us),
                "us_min": min(us), "sm_mhz": statistics.median(s["sm_mhz"] for s in samples[c]) if NVML_HANDLE else None,
                "watts": statistics.median(s["watts"] for s in samples[c]) if NVML_HANDLE else None,
                "tflops": flops / (statistics.median(us) * 1e-6) / 1e12,
            }
            results.append(row)
        del graphs
        base = next(r for r in results if r["m"] == m and r["mode"] == "dense")["us_median"]
        print(f"\nM={m} K={args.k} N={args.n} {args.dtype} (reps/graph={reps}, replays={replays})")
        print(f"{'mode':10s}{'sparsity':>9s}{'us':>10s}{'vs dense':>10s}{'TFLOPS':>9s}{'SM MHz':>8s}{'W':>7s}")
        for r in results:
            if r["m"] == m:
                print(f"{r['mode']:10s}{r['sparsity']:9.2f}{r['us_median']:10.2f}{base / r['us_median']:9.3f}x"
                      f"{r['tflops']:9.1f}{r['sm_mhz'] or 0:8.0f}{r['watts'] or 0:7.0f}", flush=True)
    gpu = torch.cuda.get_device_name(0)
    out_dir = os.path.join(args.results_dir, gpu_slug(gpu))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as f:
        json.dump({"gpu": gpu, "torch": torch.__version__, "args": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
