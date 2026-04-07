"""Classical GP mutation animation.

Generates a GIF showing 5 classical genetic programming mutations applied to
a simple Python function. Each mutation breaks the code, illustrating why
random structural mutations are ineffective for program synthesis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Colors matching the presentation CSS
BG_COLOR = "#1a1a2e"
CODE_BG = "#16213e"
BORDER_COLOR = "#4a4a6a"
WHITE = "#e0e0e0"
GREEN = "#4ade80"
PINK = "#f72585"
RED = "#ff4444"
CYAN = "#4cc9f0"
ORANGE = "#f8961e"
DIM = "#888888"

ORIGINAL_CODE = [
    "def distance(x1, y1, x2, y2):",
    "    dx = x2 - x1",
    "    dy = y2 - y1",
    "    return sqrt(dx*dx + dy*dy)",
]

MUTATIONS = [
    {
        "label": "Point Mutation",
        "description": "Replace random token",
        "code": [
            "def distance(x1, y1, x2, y2):",
            "    dx = x2 - x1",
            "    dy = y2 - y1",
            "    return [qrt(dx*dx + dy*dy)",
        ],
        "changed": [3],
        "error": "SyntaxError: invalid syntax",
    },
    {
        "label": "Subtree Crossover",
        "description": "Swap lines from another program",
        "code": [
            "    dx = x2 - x1",
            "def distance(x1, y1, x2, y2):",
            "    dy = y2 - y1",
            "    return sqrt(dx*dx + dy*dy)",
        ],
        "changed": [0, 1],
        "error": "IndentationError: unexpected indent",
    },
    {
        "label": "Subtree Mutation",
        "description": "Replace expression with random subtree",
        "code": [
            "def distance(x1, y1, x2, y2):",
            "    dx = x2 - x1",
            "    dy = y2 - y1",
            "    return sqrt(dx*dx + len([y2]))",
        ],
        "changed": [3],
        "error": "Wrong result: distance(0,0,3,4) = 3.16 (expected 5.0)",
    },
    {
        "label": "Operator Mutation",
        "description": "Replace operator with random operator",
        "code": [
            "def distance(x1, y1, x2, y2):",
            "    dx = x2 >> x1",
            "    dy = y2 - y1",
            "    return sqrt(dx*dx + dy*dy)",
        ],
        "changed": [1],
        "error": "Wrong result: distance(0,0,3,4) = 4.0 (expected 5.0)",
    },
    {
        "label": "Node Deletion",
        "description": "Remove random statement",
        "code": [
            "def distance(x1, y1, x2, y2):",
            "    dx = x2 - x1",
            "    return sqrt(dx*dx + dy*dy)",
        ],
        "changed": [],
        "error": "NameError: name 'dy' is not defined",
        "deleted_line": 2,
    },
]


def _draw_code_block(ax, lines, changed_lines, y_top, font_size=13):
    """Draw code lines with optional highlighting."""
    line_height = 0.055
    x_left = 0.08
    for i, line in enumerate(lines):
        y = y_top - i * line_height
        color = PINK if i in changed_lines else WHITE
        weight = "bold" if i in changed_lines else "normal"
        ax.text(
            x_left, y, line,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=font_size,
            color=color,
            fontweight=weight,
            verticalalignment="top",
        )


def _draw_frame(ax, stage, mutation_idx, broken_count):
    """Draw a single frame of the animation."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    if stage == "original":
        # Title
        ax.text(
            0.5, 0.95, "Original Function",
            transform=ax.transAxes, fontsize=18, color=CYAN,
            fontweight="bold", ha="center", va="top",
        )

        # Code background box
        box = mpatches.FancyBboxPatch(
            (0.04, 0.38), 0.92, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=CODE_BG, edgecolor=BORDER_COLOR, linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(box)

        _draw_code_block(ax, ORIGINAL_CODE, [], y_top=0.72)

        # Status label
        ax.text(
            0.5, 0.31, "Working",
            transform=ax.transAxes, fontsize=16, color=GREEN,
            fontweight="bold", ha="center", va="top",
        )

    elif stage == "mutation_label":
        mut = MUTATIONS[mutation_idx]
        # Title: mutation type
        ax.text(
            0.5, 0.95,
            f"Mutation {mutation_idx + 1}/5: {mut['label']}",
            transform=ax.transAxes, fontsize=18, color=ORANGE,
            fontweight="bold", ha="center", va="top",
        )
        ax.text(
            0.5, 0.88, mut["description"],
            transform=ax.transAxes, fontsize=12, color=DIM,
            ha="center", va="top",
        )

        # Code background box
        box = mpatches.FancyBboxPatch(
            (0.04, 0.38), 0.92, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=CODE_BG, edgecolor=BORDER_COLOR, linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(box)

        _draw_code_block(ax, ORIGINAL_CODE, [], y_top=0.72)

    elif stage in ("mutation_show", "mutation_hold"):
        mut = MUTATIONS[mutation_idx]
        # Title: mutation type
        ax.text(
            0.5, 0.95,
            f"Mutation {mutation_idx + 1}/5: {mut['label']}",
            transform=ax.transAxes, fontsize=18, color=ORANGE,
            fontweight="bold", ha="center", va="top",
        )
        ax.text(
            0.5, 0.88, mut["description"],
            transform=ax.transAxes, fontsize=12, color=DIM,
            ha="center", va="top",
        )

        # Code background box
        box = mpatches.FancyBboxPatch(
            (0.04, 0.38), 0.92, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=CODE_BG, edgecolor=PINK, linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(box)

        _draw_code_block(ax, mut["code"], mut["changed"], y_top=0.72)

        # Error message
        ax.text(
            0.5, 0.31, mut["error"],
            transform=ax.transAxes, fontsize=13, color=RED,
            fontweight="bold", ha="center", va="top",
            fontfamily="monospace",
        )

        # Running counter
        ax.text(
            0.5, 0.08,
            f"{broken_count}/5 mutations broke the code",
            transform=ax.transAxes, fontsize=14, color=PINK,
            fontweight="bold", ha="center", va="bottom",
        )

    elif stage == "summary":
        ax.text(
            0.5, 0.60,
            "5/5 mutations broke the code",
            transform=ax.transAxes, fontsize=24, color=RED,
            fontweight="bold", ha="center", va="center",
        )
        ax.text(
            0.5, 0.45,
            "Random structural mutations almost always",
            transform=ax.transAxes, fontsize=14, color=DIM,
            ha="center", va="center",
        )
        ax.text(
            0.5, 0.39,
            "destroy syntax or semantics",
            transform=ax.transAxes, fontsize=14, color=DIM,
            ha="center", va="center",
        )


def run(save_path=None, fps=1):
    """Generate the classical GP mutation GIF.

    Args:
        save_path: Path to save the GIF. If None, shows interactively.
        fps: Frames per second for the animation.
    """
    # Build frame sequence
    # Each entry: (stage, mutation_idx, broken_count)
    frames_data = []

    # Opening: show original code (4 frames)
    for _ in range(4):
        frames_data.append(("original", -1, 0))

    # Per mutation: label, show, hold (6 frames each)
    for i in range(5):
        broken = i  # count before this mutation
        frames_data.append(("mutation_label", i, broken))
        frames_data.append(("mutation_label", i, broken))
        frames_data.append(("mutation_show", i, broken + 1))
        frames_data.append(("mutation_show", i, broken + 1))
        frames_data.append(("mutation_hold", i, broken + 1))
        frames_data.append(("mutation_hold", i, broken + 1))

    # Summary (8 frames)
    for _ in range(8):
        frames_data.append(("summary", -1, 5))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update(frame_idx):
        stage, mut_idx, broken_count = frames_data[frame_idx]
        _draw_frame(ax, stage, mut_idx, broken_count)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames_data), interval=1000 // fps, blit=False,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    figures = Path(__file__).parent / "figures"
    run(save_path=str(figures / "classical_gp.gif"))
