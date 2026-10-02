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


if __name__ == "__main__":
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*", "*.json"))):
        print(plot_run(p))
