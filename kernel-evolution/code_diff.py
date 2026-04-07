"""Side-by-side code diff image for the presentation.

Generates a PNG showing the original naive softmax kernel next to the
Hopper TMA-accelerated version, with key additions highlighted.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Colors matching the presentation CSS
BG_COLOR = "#1a1a2e"
CODE_BG = "#16213e"
BORDER_COLOR = "#4a4a6a"
WHITE = "#e0e0e0"
CYAN = "#4cc9f0"
PINK = "#f72585"
GREEN = "#4ade80"
ORANGE = "#f8961e"
DIM = "#888888"

# Trimmed original kernel — just the essential body
ORIGINAL_LINES = [
    ("block_id = cute.arch.block_idx()[0]", None),
    ("dens = cute.Float32(0.0)", None),
    ('max_val = cast(float("-inf"), dtype)', None),
    ("", None),
    ("for tile_idx in cutlass.range(num_tiles):", None),
    ("    x = input[block_id, tile_idx].load()", None),
    ("    local_max = x.reduce(MAX)", None),
    ("    next_max = max(max_val, local_max)", None),
    ("    dens = dens * exp(max_val - next_max)", None),
    ("    s = exp(x - next_max).reduce(ADD)", None),
    ("    dens += s", None),
    ("    max_val = next_max", None),
    ("", None),
    ("for tile_idx in cutlass.range(num_tiles):", None),
    ("    x = input[block_id, tile_idx].load()", None),
    ("    output[...] = exp(x - max_val) / dens", None),
]

# Trimmed Hopper kernel — showing TMA + SMEM additions
HOPPER_LINES = [
    ("# Swizzled SMEM (WGMMA-compatible)", CYAN),
    ("smem = SmemAllocator()", CYAN),
    ("s_in = smem.allocate_tensor(dtype,", CYAN),
    ("    layout, swizzle=K_SW128)", CYAN),
    ("mbar = smem.allocate_array(Uint64, 1)", CYAN),
    ("", None),
    ("# Barrier init (elect one thread)", PINK),
    ("with elect_one():", PINK),
    ("    mbarrier_init(mbar, 1)", PINK),
    ("", None),
    ("for tile_k in cutlass.range(num_tiles):", None),
    ("    # Async TMA: global -> shared", ORANGE),
    ("    mbarrier_arrive_and_expect_tx(mbar)", ORANGE),
    ("    cute.copy(tma_atom, gS, sS,", ORANGE),
    ("             tma_bar_ptr=mbar)", ORANGE),
    ("    mbarrier_wait(mbar, 0)", ORANGE),
    ("    fence_proxy(async_shared)", ORANGE),
    ("", None),
    ("    val = s_flat[elem_idx]", None),
    ("    new_max = max(old_max, val)", None),
    ("    denoms[i] = denoms[i]*exp(old-new)", None),
    ("               + exp(val - new_max)", None),
]


def run(save_path=None):
    """Generate the side-by-side code diff PNG.

    Args:
        save_path: Path to save the PNG. If None, shows interactively.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor(BG_COLOR)

    for ax, title, lines, border_color in [
        (ax_left, "Before", ORIGINAL_LINES, BORDER_COLOR),
        (ax_right, "After", HOPPER_LINES, CYAN),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")

        # Title
        title_color = DIM if title == "Before" else CYAN
        ax.text(
            0.5, 0.97, title,
            transform=ax.transAxes, fontsize=15, color=title_color,
            fontweight="bold", ha="center", va="top",
        )

        # Code background box
        box = mpatches.FancyBboxPatch(
            (0.03, 0.02), 0.94, 0.88,
            boxstyle="round,pad=0.015",
            facecolor=CODE_BG, edgecolor=border_color,
            linewidth=1.5 if title == "Before" else 2.0,
            transform=ax.transAxes,
        )
        ax.add_patch(box)

        # Code lines
        line_height = 0.038
        y_top = 0.87
        x_left = 0.07

        for i, (text, color) in enumerate(lines):
            y = y_top - i * line_height
            line_color = color if color else WHITE
            ax.text(
                x_left, y, text,
                transform=ax.transAxes,
                fontfamily="monospace",
                fontsize=8.5,
                color=line_color,
                verticalalignment="top",
            )

    # Legend at the bottom
    legend_items = [
        (CYAN, "Shared memory"),
        (PINK, "Barrier sync"),
        (ORANGE, "TMA async loads"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x_pos = 0.28 + i * 0.22
        fig.text(
            x_pos, 0.015, f"  {label}",
            fontsize=9, color=color, fontweight="bold",
            va="center",
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=180, facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    figures = Path(__file__).parent / "figures"
    run(save_path=str(figures / "code_diff.png"))
