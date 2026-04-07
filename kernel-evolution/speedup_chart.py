"""Speedup chart for custom kernel vs torch.compile on H100."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# Colors matching the presentation CSS
BG_COLOR = "#1a1a2e"
CODE_BG = "#16213e"
BORDER_COLOR = "#4a4a6a"
WHITE = "#e0e0e0"
CYAN = "#4cc9f0"
PINK = "#f72585"
GREEN = "#4ade80"
DIM = "#888888"

# LRKV fused_lorak_rope H100 benchmark data (median latency in ms)
SHAPE_LABELS = [
    "8×512",
    "16×1K",
    "32×2K",
    "64×4K",
    "128×8K",
]

TORCH_MEDIAN = [0.08675, 0.31891, 1.16502, 4.55110, 18.12768]
CUTE_MEDIAN = [0.05366, 0.17678, 0.68614, 2.67952, 10.65290]

SPEEDUPS = [t / c for t, c in zip(TORCH_MEDIAN, CUTE_MEDIAN)]


def run(save_path=None):
    """Generate the speedup chart.

    Args:
        save_path: Path to save the PNG. If None, shows interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CODE_BG)

    x = np.arange(len(SHAPE_LABELS))
    bar_colors = [PINK if s < 1.0 else CYAN for s in SPEEDUPS]

    bars = ax.bar(x, SPEEDUPS, width=0.6, color=bar_colors, edgecolor="none",
                  alpha=0.9, zorder=3)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, SPEEDUPS)):
        label = f"{speedup:.2f}x"
        y_pos = bar.get_height() + 0.03
        color = PINK if speedup < 1.0 else WHITE
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos, label,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=color,
        )

    # Reference line at 1.0x
    ax.axhline(y=1.0, color=DIM, linestyle="--", linewidth=1.2, zorder=2,
               alpha=0.7)
    ax.text(
        len(SHAPE_LABELS) - 0.5, 1.03, "torch.compile baseline",
        color=DIM, fontsize=9, ha="right", va="bottom",
    )

    # Styling
    ax.set_xlabel("Input Shape (batch × seq_len)", color=WHITE, fontsize=12,
                  labelpad=10)
    ax.set_ylabel("Speedup vs torch.compile", color=WHITE, fontsize=12,
                  labelpad=10)
    ax.set_title(
        "Kernel for Custom Architecture — H100\nSpeedup vs torch.compile",
        color=WHITE, fontsize=16, fontweight="bold", pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(SHAPE_LABELS, color=WHITE, fontsize=11)
    ax.tick_params(axis="y", colors=WHITE, labelsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    ax.grid(axis="y", color=BORDER_COLOR, alpha=0.3, zorder=1)
    ax.set_ylim(0, max(SPEEDUPS) + 0.35)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    edgecolor="none")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    figures = Path(__file__).parent / "figures"
    run(save_path=str(figures / "speedup_chart.png"))
