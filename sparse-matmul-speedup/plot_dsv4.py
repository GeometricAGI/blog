"""Plot the DeepSeek-V4-Flash-shaped sweep: speedup vs. sparsity, one panel per weight shape."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#d6602a", "#2a9d6e", "#8a5ad6", "#6b6b66", "#b8a21e", "#c23b7a"]


def plot_run(path: str) -> str:
    """Render one ``results/<gpu>/dsv4-flash/<tag>.json`` file.

    Args:
        path: Path to a results file written with ``--suite dsv4-flash``.

    Returns:
        Path of the PNG written next to the JSON file.
    """
    data = json.load(open(path))
    rows = data["results"]
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.4 * len(shapes), 4.2), dpi=160, sharey=True)
    for ax, shape in zip(axes, shapes):
        sel = [r for r in rows if r["shape"] == shape]
        k, n = sel[0]["k"], sel[0]["n"]
        for color, tokens in zip(COLORS, sorted({r["tokens"] for r in sel})):
            cur = [r for r in sel if r["tokens"] == tokens]
            base = next(r for r in cur if r["mode"] == "dense")
            pts = [(0.0, 1.0)] + sorted((r["sparsity"], base["us_median"] / r["us_median"]) for r in cur if r["mode"] == "random")
            ax.plot(*zip(*pts), marker="o", ms=3.5, lw=1.8, color=color, label=f"{tokens} tok ({base['m']} rows)")
        ax.axhline(1.0, color="#999", lw=1, ls="--")
        ax.set_title(f"{shape} ({k}->{n})", loc="left", fontsize=10)
        ax.set_xlabel("fraction of weight zeroed")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("speedup vs. dense weight")
    axes[0].set_ylim(0.92, 1.25)
    fig.suptitle(f"{data['gpu']} - {os.path.basename(path)[:-5]}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = path[:-5] + ".png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    for p in sorted(glob.glob(os.path.join(root, "*", "dsv4-flash", "*.json"))):
        print(plot_run(p))
