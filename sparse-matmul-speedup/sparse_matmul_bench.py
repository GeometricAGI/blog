"""Does a dense matmul get faster when one operand is sparse (zeros stored densely)?

Times dense ``torch.matmul`` replayed through a CUDA graph with the weight zeroed to
different degrees. Also logs SM clock and power so a speedup can be attributed to
data-dependent power/clock effects rather than skipped work.

Each graph rotates through independent copies of the weight, activations and output, so
consecutive uses of the same tensor are separated by far more than the L2 capacity.

Weights are stored as ``(out_features, in_features)`` like ``nn.Linear`` and applied as
``x @ w.t()``. Shapes are given as ``name:in:out[:row_fraction]``; ``row_fraction`` scales the
token count to the rows each matmul actually sees (e.g. 6/256 for a routed MoE expert).
"""

import argparse
import json
import math
import os
import re
import statistics

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    # Look the device up by UUID: NVML indexes physical GPUs and ignores CUDA_VISIBLE_DEVICES.
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{torch.cuda.get_device_properties(0).uuid}")
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


def make_weight(n: int, k: int, sparsity: float, mode: str, dtype: torch.dtype) -> torch.Tensor:
    """Build an ``n x k`` (out x in) weight with a given fraction of entries zeroed.

    Args:
        n: Output features (rows of the weight).
        k: Input features (the contraction dimension).
        sparsity: Fraction of entries set to zero, in [0, 1).
        mode: ``random`` zeroes exactly ``round(sparsity * n * k)`` entries chosen
            uniformly at random, ``magnitude`` zeroes the smallest-magnitude entries.
        dtype: Output dtype.

    Returns:
        A dense ``(n, k)`` CUDA tensor containing the zeros.
    """
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    num_zero = round(sparsity * w.numel())
    if num_zero > 0:
        if mode == "random":
            zero_idx = torch.randperm(w.numel(), device="cuda")[:num_zero]
            w.view(-1)[zero_idx] = 0
        else:
            thresh = w.abs().flatten().kthvalue(num_zero).values
            # torch.where gives +0.0; multiplying by a mask would leave -0.0 for negative weights
            w = torch.where(w.abs() > thresh, w, 0.0)
    return w.to(dtype).contiguous()


def capture(xs: list, ws: list, outs: list, reps: int):
    """Capture ``reps`` back-to-back matmuls into one CUDA graph, rotating through the copies.

    Matmul ``i`` uses ``xs[i % len(xs)] @ ws[i % len(ws)].t()``, so the same tensors are only
    touched again after ``len(xs) - 1`` other matmuls' worth of traffic.

    Args:
        xs: Activations, each of shape ``(m, k)``.
        ws: Weights, each of shape ``(n, k)`` (applied as ``x @ w.t()``); same length as ``xs``.
        outs: Output buffers, each of shape ``(m, n)``; same length as ``xs``.
        reps: Matmuls per graph, a multiple of ``len(xs)``.

    Returns:
        The captured ``torch.cuda.CUDAGraph``.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(3):
            torch.matmul(xs[i % len(xs)], ws[i % len(ws)].t(), out=outs[i % len(outs)])
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            torch.matmul(xs[i % len(xs)], ws[i % len(ws)].t(), out=outs[i % len(outs)])
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


def parse_shape(spec: str):
    """Parse a ``name:in:out[:row_fraction]`` shape spec.

    Args:
        spec: Shape spec, e.g. ``expert_gate_up:4096:4096:0.0234375``.

    Returns:
        ``(name, k, n, row_fraction)`` with ``k`` input and ``n`` output features.
    """
    parts = spec.split(":")
    return parts[0], int(parts[1]), int(parts[2]), float(parts[3]) if len(parts) > 3 else 1.0


def main() -> None:
    """Run the sweep and write one results JSON per run."""
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["square8192:8192:8192"],
                   help="name:in:out[:row_fraction] weight shapes.")
    p.add_argument("--ms", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384], help="Token counts.")
    p.add_argument("--sparsities", type=float, nargs="+", default=[0, 0.05, 0.10, 0.25, 0.50, 0.70, 0.90, 0.99])
    p.add_argument("--modes", nargs="+", default=["random", "magnitude"])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--copies", type=int, default=0,
                   help="Copies rotated through per graph; 0 picks enough for --rotate-mib of weight traffic.")
    p.add_argument("--rotate-mib", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tag", default="unlocked", help="Run label, e.g. 'unlocked' or 'locked-1200mhz'.")
    p.add_argument("--results-dir", default="results", help="Results go to <results-dir>/<gpu>/[<suite>/]<tag>.json.")
    p.add_argument("--suite", default="", help="Optional subdirectory under the GPU directory.")
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    # configs: (mode, sparsity), with one dense baseline.
    configs = [("dense", 0.0)] + [(m, s) for m in args.modes for s in args.sparsities if s > 0]
    results = []
    for spec in args.shapes:
        name, k, n, frac = parse_shape(spec)
        w_bytes = n * k * torch.empty((), dtype=dtype).element_size()
        copies = args.copies or min(64, max(2, math.ceil(args.rotate_mib * 2**20 / w_bytes)))
        weights, zero_frac = {}, {}
        for mode, s in configs:
            weights[(mode, s)] = [
                make_weight(n, k, s, "random" if mode == "dense" else mode, dtype) for _ in range(copies)
            ]
            fracs = [(w == 0).sum().item() / w.numel() for w in weights[(mode, s)]]
            zero_frac[(mode, s)] = statistics.mean(fracs)
            print(f"[{name}] built {mode} {s} x{copies}: zero frac {min(fracs):.6f}..{max(fracs):.6f}", flush=True)

        seen_rows = set()
        for tokens in args.ms:
            m = max(1, math.ceil(tokens * frac))
            if m in seen_rows:
                continue
            seen_rows.add(m)
            xs = [torch.randn(m, k, device="cuda", dtype=dtype) for _ in range(copies)]
            outs = [torch.empty(m, n, device="cuda", dtype=dtype) for _ in range(copies)]
            # keep the wall time per shape ~constant, but always whole passes over the copies
            flops = 2 * m * k * n
            budget = max(1, min(50, int(2e12 / flops)))
            reps = copies * -(-budget // copies)
            replays = max(5, min(200, int(4e13 / (flops * reps))))
            graphs = {c: capture(xs, weights[c], outs, reps) for c in configs}
            samples = {c: [] for c in configs}
            for r in range(args.rounds):
                # alternate order each round so thermal/clock drift hits every config equally
                order = configs if r % 2 == 0 else configs[::-1]
                for c in order:
                    samples[c].append(time_graph(graphs[c], replays, reps))
            first = len(results)
            for c in configs:
                us = [smp["us"] for smp in samples[c]]
                results.append({
                    "shape": name, "k": k, "n": n, "tokens": tokens, "m": m, "mode": c[0], "sparsity": c[1],
                    "copies": copies, "us_median": statistics.median(us), "us_min": min(us),
                    "zero_frac": zero_frac[c],
                    "sm_mhz": statistics.median(smp["sm_mhz"] for smp in samples[c]) if NVML_HANDLE else None,
                    "watts": statistics.median(smp["watts"] for smp in samples[c]) if NVML_HANDLE else None,
                    "tflops": flops / (statistics.median(us) * 1e-6) / 1e12,
                })
            del graphs, xs, outs
            base = next(r for r in results[first:] if r["mode"] == "dense")["us_median"]
            print(f"\n{name} tokens={tokens} rows={m} K={k} N={n} {args.dtype} "
                  f"(copies={copies}, reps/graph={reps}, replays={replays})")
            print(f"{'mode':10s}{'sparsity':>9s}{'us':>10s}{'vs dense':>10s}{'TFLOPS':>9s}{'SM MHz':>8s}{'W':>7s}")
            for r in results[first:]:
                print(f"{r['mode']:10s}{r['sparsity']:9.2f}{r['us_median']:10.2f}{base / r['us_median']:9.3f}x"
                      f"{r['tflops']:9.1f}{r['sm_mhz'] or 0:8.0f}{r['watts'] or 0:7.0f}", flush=True)
        del weights
        torch.cuda.empty_cache()
    gpu = torch.cuda.get_device_name(0)
    out_dir = os.path.join(args.results_dir, gpu_slug(gpu), args.suite)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}.json"), "w") as f:
        json.dump({"gpu": gpu, "torch": torch.__version__, "args": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
