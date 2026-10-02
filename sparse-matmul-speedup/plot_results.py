"""Plot speedup-vs-dense for every ``results/<gpu>/<tag>.json`` into ``results/<gpu>/<tag>.png``."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#d6602a", "#2a9d6e", "#8a5ad6", "#6b6b66"]


def plot_run(path: str) -> str:
    """Render one results file as speedup vs. sparsity, one line per M.

    Args:
        path: Path to a ``results/<gpu>/<tag>.json`` file written by the benchmark.

    Returns:
        Path of the PNG written next to the JSON file.
    """
    data = json.load(open(path))
    rows = data["results"]
    ms = sorted({r["m"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    for color, m in zip(COLORS, ms):
        base = next(r for r in rows if r["m"] == m and r["mode"] == "dense")["us_median"]
        pts = sorted((r["sparsity"], base / r["us_median"]) for r in rows if r["m"] == m and r["mode"] == "random")
        pts = [(0.0, 1.0)] + pts
        ax.plot(*zip(*pts), marker="o", ms=4, lw=2, color=color, label=f"M={m}")
    ax.axhline(1.0, color="#999", lw=1, ls="--")
    ax.set_xlabel("fraction of weight entries set to zero (random)")
    ax.set_ylabel("speedup vs. dense weight")
    ax.set_ylim(0.9, 1.2)
    ax.set_title(f"{data['gpu']} - {os.path.basename(path)[:-5]}", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = path[:-5] + ".png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_clock_sweep(gpu_dir: str, sparsity: float = 0.99):
    """Plot speedup at one sparsity level against the locked and the achieved SM clock.

    Left panel: x is the requested lock. Right panel: x is the SM clock the dense run
    actually reached (NVML sample), which is lower than the lock once the GPU power-throttles.

    Args:
        gpu_dir: A ``results/<gpu>`` directory holding ``locked-<mhz>mhz.json`` files.
        sparsity: Random-zeroing level to plot.

    Returns:
        Path of the ``clock_sweep.png`` written in ``gpu_dir``, or ``None`` if the
        directory has no locked-clock runs.
    """
    runs = {}
    for path in glob.glob(os.path.join(gpu_dir, "locked-*mhz.json")):
        runs[int(os.path.basename(path)[7:-8])] = json.load(open(path))
    if not runs:
        return None
    gpu = next(iter(runs.values()))["gpu"]
    fig, (ax_lock, ax_real) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, sharey=True)
    for color, m in zip(COLORS, sorted({r["m"] for r in next(iter(runs.values()))["results"]})):
        locked, achieved, speedup = [], [], []
        for mhz, data in sorted(runs.items()):
            rows = [r for r in data["results"] if r["m"] == m]
            dense = next(r for r in rows if r["mode"] == "dense")
            sparse = next(r for r in rows if r["mode"] == "random" and r["sparsity"] == sparsity)
            locked.append(mhz)
            achieved.append(dense["sm_mhz"])
            speedup.append(dense["us_median"] / sparse["us_median"])
        ax_lock.plot(locked, speedup, marker="o", ms=4, lw=2, color=color, label=f"M={m}")
        ax_real.scatter(achieved, speedup, s=22, color=color)
    for ax, xlabel in ((ax_lock, "locked SM clock (MHz)"), (ax_real, "SM clock the dense run reached (MHz)")):
        ax.axhline(1.0, color="#999", lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    ax_lock.set_ylabel(f"speedup vs. dense weight ({sparsity:.0%} zeros)")
    ax_lock.set_title(f"{gpu} - by requested lock", loc="left", fontsize=11)
    ax_real.set_title("by achieved clock", loc="left", fontsize=11)
    ax_lock.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(gpu_dir, "clock_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    for p in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        print(plot_run(p))
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        out = plot_clock_sweep(d)
        if out:
            print(out)
