"""Generate static figures for the 'slightly deeper' version of the diversity post.

Figures:
1. phase_diagram.png        — Selection pressure vs mutation strength heatmap
2. time_series.png          — Best fitness + diversity over generations (3 regimes)
3. takeover_curves.png      — Takeover dynamics for different tournament sizes
4. mutation_phase_transition.png — e^c/c curve showing optimal mutation rate
5. self_adjusting_onemax.png — Self-adjusting mutation rate on OneMax
6. markov_chain.png         — Markov chain: without vs with elitism
7. crossover_cartoon.png    — Crossover with diverse vs identical parents
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---------- shared styling ----------

DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
ACCENT_BLUE = "#4cc9f0"
ACCENT_GREEN = "#4ade80"
ACCENT_RED = "#f72585"
ACCENT_ORANGE = "#f8961e"
ACCENT_PURPLE = "#b388ff"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.size": 11,
})


def rastrigin(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))


# =====================================================================
# Figure 1: Phase diagram
# =====================================================================

def _run_ea_final_fitness(tourn_size, sigma, pop_size=30, generations=80,
                          bound=5.12, seed=42):
    """Run EA and return best fitness at the end."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-bound, bound, (pop_size, 2))
    fit = rastrigin(pop[:, 0], pop[:, 1])

    for gen in range(generations):
        new_pop = np.empty_like(pop)
        k = min(int(tourn_size), pop_size)
        for j in range(pop_size):
            candidates = rng.choice(pop_size, k, replace=False)
            winner = candidates[np.argmin(fit[candidates])]
            new_pop[j] = pop[winner]
        mutation = rng.normal(0, sigma, (pop_size, 2))
        new_pop = np.clip(new_pop + mutation, -bound, bound)
        # Elitism
        best_idx = np.argmin(fit)
        new_fit = rastrigin(new_pop[:, 0], new_pop[:, 1])
        worst_new = np.argmax(new_fit)
        new_pop[worst_new] = pop[best_idx]
        pop = new_pop
        fit = rastrigin(pop[:, 0], pop[:, 1])

    return fit.min()


def make_phase_diagram(save_path):
    tourn_sizes = np.array([2, 3, 4, 5, 7, 10, 15, 20])
    sigmas = np.logspace(np.log10(0.05), np.log10(5.0), 16)

    # Average over a few seeds for smoother results
    seeds = [42, 123, 456]
    grid = np.zeros((len(sigmas), len(tourn_sizes)))
    for si, sigma in enumerate(sigmas):
        for ti, ts in enumerate(tourn_sizes):
            vals = [_run_ea_final_fitness(ts, sigma, seed=s) for s in seeds]
            grid[si, ti] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.arange(len(tourn_sizes) + 1) - 0.5,
        np.arange(len(sigmas) + 1) - 0.5,
        grid, cmap="inferno_r", shading="flat",
    )
    ax.set_xticks(range(len(tourn_sizes)))
    ax.set_xticklabels(tourn_sizes)
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f"{s:.2f}" for s in sigmas], fontsize=8)
    ax.set_xlabel("Tournament size (selection pressure →)", fontsize=12)
    ax.set_ylabel("Mutation σ (exploration →)", fontsize=12)
    ax.set_title("Final best fitness on Rastrigin (lower = better)", fontsize=13,
                 fontweight="bold", color="white")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Best fitness after 80 generations", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    # Annotate regions
    ax.text(0.5, len(sigmas) - 1.5, "Random search\n(too much mutation)",
            color=ACCENT_ORANGE, fontsize=9, fontweight="bold", ha="left")
    ax.text(len(tourn_sizes) - 1.5, 0.5, "Premature\nconvergence",
            color=ACCENT_RED, fontsize=9, fontweight="bold", ha="right")
    ax.text(len(tourn_sizes) // 2 - 0.5, len(sigmas) // 2,
            "Useful\nsearch", color=ACCENT_GREEN, fontsize=10,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                      edgecolor=ACCENT_GREEN, alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 2: Time series (best fitness + diversity for 3 regimes)
# =====================================================================

def _run_ea_history(tourn_size, sigma_or_schedule, pop_size=30, generations=80,
                    bound=5.12, seed=42):
    """Run EA and return per-generation best fitness and mean pairwise distance."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-bound, bound, (pop_size, 2))
    fit = rastrigin(pop[:, 0], pop[:, 1])

    best_hist = []
    div_hist = []

    for gen in range(generations):
        best_hist.append(fit.min())
        # Mean pairwise distance
        diffs = pop[:, None, :] - pop[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        div_hist.append(dists.sum() / (pop_size * (pop_size - 1)))

        new_pop = np.empty_like(pop)
        k = min(int(tourn_size), pop_size)
        for j in range(pop_size):
            candidates = rng.choice(pop_size, k, replace=False)
            winner = candidates[np.argmin(fit[candidates])]
            new_pop[j] = pop[winner]

        if callable(sigma_or_schedule):
            sigma = sigma_or_schedule(gen, generations)
        else:
            sigma = sigma_or_schedule
        mutation = rng.normal(0, sigma, (pop_size, 2))
        new_pop = np.clip(new_pop + mutation, -bound, bound)

        best_idx = np.argmin(fit)
        new_fit = rastrigin(new_pop[:, 0], new_pop[:, 1])
        worst_new = np.argmax(new_fit)
        new_pop[worst_new] = pop[best_idx]
        pop = new_pop
        fit = rastrigin(pop[:, 0], pop[:, 1])

    best_hist.append(fit.min())
    diffs = pop[:, None, :] - pop[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    div_hist.append(dists.sum() / (pop_size * (pop_size - 1)))

    return np.array(best_hist), np.array(div_hist)


def make_time_series(save_path):
    configs = [
        ("Low diversity", 10, 0.15, ACCENT_RED),
        ("High diversity", 2, 3.5, ACCENT_ORANGE),
        ("Balanced", 5, lambda g, G: 1.2 * (1 - 0.8 * g / G), ACCENT_GREEN),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

    for col, (label, ts, sigma, color) in enumerate(configs):
        best, div = _run_ea_history(ts, sigma)
        gens = np.arange(len(best))

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(gens, best, color=color, linewidth=2)
        ax_top.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax_top.set_ylabel("Best fitness" if col == 0 else "")
        ax_top.grid(True)
        ax_top.set_ylim(-1, max(best) * 1.1)

        ax_bot.plot(gens, div, color=color, linewidth=2, linestyle="--")
        ax_bot.set_ylabel("Mean pairwise dist" if col == 0 else "")
        ax_bot.set_xlabel("Generation")
        ax_bot.grid(True)

    fig.suptitle("Fitness and Diversity Over Time", fontsize=14,
                 fontweight="bold", color="white", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 3: Takeover curves
# =====================================================================

def make_takeover_curves(save_path):
    pop_size = 30
    max_iters = 300
    n_trials = 200
    tournament_sizes = [2, 3, 5, 10]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED]

    fig, ax = plt.subplots(figsize=(8, 5))

    for ts, color in zip(tournament_sizes, colors):
        fraction_curves = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial)
            # Track how many copies of the "best" type exist
            # Start with 1 best individual out of pop_size
            types = np.zeros(pop_size, dtype=int)
            types[0] = 1  # individual 0 is "best"
            fracs = [1.0 / pop_size]

            for it in range(max_iters):
                new_types = np.empty_like(types)
                for j in range(pop_size):
                    candidates = rng.choice(pop_size, min(ts, pop_size), replace=False)
                    # Best type (1) wins tournament
                    if np.any(types[candidates] == 1):
                        new_types[j] = 1
                    else:
                        new_types[j] = 0
                types = new_types
                fracs.append(np.mean(types == 1))
                if np.all(types == 1):
                    # Pad rest with 1.0
                    fracs.extend([1.0] * (max_iters - it - 1))
                    break

            fraction_curves.append(fracs[:max_iters + 1])

        mean_frac = np.mean(fraction_curves, axis=0)
        ax.plot(range(len(mean_frac)), mean_frac, color=color, linewidth=2,
                label=f"Tournament size {ts}")

    # Theoretical takeover time for binary tournament: n * H_{n-1}
    H = sum(1.0 / k for k in range(1, pop_size))
    theo_binary = pop_size * H
    ax.axvline(theo_binary, color=ACCENT_BLUE, linestyle=":", alpha=0.7)
    ax.text(theo_binary + 2, 0.5, f"E[T]={theo_binary:.0f}\n(binary, theory)",
            color=ACCENT_BLUE, fontsize=9)

    ax.set_xlabel("Selection iterations (no mutation)", fontsize=12)
    ax.set_ylabel("Fraction of 'best' type in population", fontsize=12)
    ax.set_title("Takeover Dynamics by Tournament Size", fontsize=13,
                 fontweight="bold", color="white")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=10)
    ax.set_xlim(0, 200)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 4: Mutation phase transition (e^c / c curve)
# =====================================================================

def make_mutation_phase_transition(save_path):
    c = np.linspace(0.1, 5.0, 500)
    f = np.exp(c) / c

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(c, f, color=ACCENT_BLUE, linewidth=2.5)

    # Mark minimum near c=1
    c_opt = c[np.argmin(f)]
    f_opt = np.min(f)
    ax.plot(c_opt, f_opt, "o", color=ACCENT_GREEN, markersize=10, zorder=5)
    ax.annotate(f"  Optimum: c ≈ {c_opt:.2f}\n  (p ≈ 1/n)",
                xy=(c_opt, f_opt), xytext=(c_opt + 0.8, f_opt + 3),
                color=ACCENT_GREEN, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5))

    # Mark phase transition region
    c_thresh = np.log(50)  # ln(n) for n=50 as illustration
    ax.axvspan(c_thresh, 5.0, alpha=0.15, color=ACCENT_RED)
    ax.text(4.2, 15, "Superpolynomial\nregime", color=ACCENT_RED,
            fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("c  (mutation rate p = c/n)", fontsize=12)
    ax.set_ylabel("Leading constant  e$^c$/c  in runtime", fontsize=12)
    ax.set_title("Mutation Rate Phase Transition (Witt, 2013)", fontsize=13,
                 fontweight="bold", color="white")
    ax.set_ylim(0, 35)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 5: Self-adjusting mutation rate on OneMax
# =====================================================================

def make_self_adjusting_onemax(save_path):
    n = 200
    lam = 10  # lambda
    max_evals = 15000
    rng = np.random.default_rng(42)

    # Initial bitstring
    x = rng.integers(0, 2, size=n)
    fitness = x.sum()

    rate = 1.0 / n  # initial mutation rate
    rate_history = [rate]
    fitness_history = [fitness]
    eval_count = 0

    while fitness < n and eval_count < max_evals:
        # Two subpopulations: half with rate*2, half with rate/2
        rate_high = min(rate * 2, 0.5)
        rate_low = max(rate / 2, 0.5 / n)

        best_offspring = None
        best_fit = -1
        best_from_high = False

        for i in range(lam):
            r = rate_high if i < lam // 2 else rate_low
            # Mutate: flip each bit with probability r
            mask = rng.random(n) < r
            child = x.copy()
            child[mask] = 1 - child[mask]
            child_fit = child.sum()
            eval_count += 1

            if child_fit > best_fit:
                best_fit = child_fit
                best_offspring = child
                best_from_high = (i < lam // 2)

        # Accept if at least as good (elitist)
        if best_fit >= fitness:
            x = best_offspring
            fitness = best_fit

        # Update rate based on which subpopulation won
        if best_from_high:
            rate = min(rate * 2, 0.5)
        else:
            rate = max(rate / 2, 0.5 / n)

        rate_history.append(rate)
        fitness_history.append(fitness)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    steps = range(len(fitness_history))
    ax1.plot(steps, fitness_history, color=ACCENT_GREEN, linewidth=1.5)
    ax1.set_ylabel("Fitness (OneMax)", fontsize=11)
    ax1.axhline(n, color=ACCENT_GREEN, linestyle=":", alpha=0.4)
    ax1.set_title("Self-Adjusting (1+λ) EA on OneMax", fontsize=13,
                  fontweight="bold", color="white")
    ax1.grid(True)

    ax2.semilogy(steps, rate_history, color=ACCENT_PURPLE, linewidth=1.5)
    ax2.axhline(1.0 / n, color=ACCENT_BLUE, linestyle=":", alpha=0.6)
    ax2.text(len(steps) * 0.7, 1.2 / n, "p = 1/n", color=ACCENT_BLUE, fontsize=9)
    ax2.set_ylabel("Mutation rate p", fontsize=11)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.grid(True)

    # Annotate behaviour
    # Find a stagnation phase (fitness plateau)
    fit_arr = np.array(fitness_history)
    diffs = np.diff(fit_arr)
    # Find longest run of zeros
    stag_start = None
    stag_len = 0
    cur_start = 0
    cur_len = 0
    for i, d in enumerate(diffs):
        if d == 0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > stag_len:
                stag_len = cur_len
                stag_start = cur_start
            cur_len = 0
    if cur_len > stag_len:
        stag_len = cur_len
        stag_start = cur_start

    if stag_start is not None and stag_len > 5:
        mid = stag_start + stag_len // 2
        ax2.annotate("Stagnation →\nrate increases",
                     xy=(mid, rate_history[mid]),
                     xytext=(mid + len(steps) * 0.1, rate_history[mid] * 5),
                     color=ACCENT_ORANGE, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.5))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 6: Markov chain diagram (without vs with elitism)
# =====================================================================

def make_markov_chain(save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax in (ax1, ax2):
        ax.set_xlim(-1.5, 4.5)
        ax.set_ylim(-1.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

    def draw_state(ax, x, y, label, color, bold=False):
        circle = plt.Circle((x, y), 0.5, facecolor=PANEL_BG, edgecolor=color,
                             linewidth=2.5 if bold else 1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal", color=color, zorder=6)

    def draw_arrow(ax, x1, y1, x2, y2, color, curved=0):
        style = f"arc3,rad={curved}" if curved != 0 else "arc3,rad=0.0"
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="->,head_width=6,head_length=6",
                                color=color, linewidth=1.5,
                                connectionstyle=style, zorder=4)
        ax.add_patch(arrow)

    def draw_self_loop(ax, x, y, color, direction="top"):
        if direction == "top":
            loop = FancyArrowPatch((x - 0.3, y + 0.48), (x + 0.3, y + 0.48),
                                   arrowstyle="->,head_width=5,head_length=5",
                                   color=color, linewidth=1.3,
                                   connectionstyle="arc3,rad=-1.5", zorder=4)
        else:
            loop = FancyArrowPatch((x - 0.3, y - 0.48), (x + 0.3, y - 0.48),
                                   arrowstyle="->,head_width=5,head_length=5",
                                   color=color, linewidth=1.3,
                                   connectionstyle="arc3,rad=1.5", zorder=4)
        ax.add_patch(loop)

    # ---- Left panel: WITHOUT elitism (ergodic, no absorbing state) ----
    ax1.set_title("Without Elitism", fontsize=13, fontweight="bold",
                  color=ACCENT_RED, pad=15)

    states_left = [
        (0, 0.5, "Sub-\noptimal", TEXT_COLOR),
        (2, 1.5, "Near\noptimum", TEXT_COLOR),
        (3.5, 0.5, "Optimum\nfound", ACCENT_ORANGE),
    ]
    for x, y, label, color in states_left:
        draw_state(ax1, x, y, label, color)

    # Arrows: everything connects to everything (ergodic)
    draw_arrow(ax1, 0.5, 0.7, 1.5, 1.3, TEXT_COLOR, curved=0.2)
    draw_arrow(ax1, 1.5, 1.3, 0.5, 0.7, TEXT_COLOR, curved=0.2)
    draw_arrow(ax1, 2.5, 1.3, 3.0, 0.7, TEXT_COLOR, curved=0.2)
    draw_arrow(ax1, 3.0, 0.7, 2.5, 1.3, ACCENT_RED, curved=0.2)  # can LEAVE optimum
    draw_arrow(ax1, 0.45, 0.15, 3.05, 0.15, TEXT_COLOR, curved=-0.3)
    draw_self_loop(ax1, 0, 0.5, TEXT_COLOR, "bottom")

    ax1.text(3.2, 1.5, "Can leave\noptimum!", color=ACCENT_RED,
             fontsize=9, fontweight="bold", fontstyle="italic")

    # ---- Right panel: WITH elitism (absorbing state) ----
    ax2.set_title("With Elitism", fontsize=13, fontweight="bold",
                  color=ACCENT_GREEN, pad=15)

    states_right = [
        (0, 0.5, "Sub-\noptimal", TEXT_COLOR),
        (2, 1.5, "Near\noptimum", TEXT_COLOR),
        (3.5, 0.5, "Optimum\nfound", ACCENT_GREEN),
    ]
    for x, y, label, color in states_right:
        draw_state(ax2, x, y, label, color, bold=(label == "Optimum\nfound"))

    draw_arrow(ax2, 0.5, 0.7, 1.5, 1.3, TEXT_COLOR, curved=0.2)
    draw_arrow(ax2, 1.5, 1.3, 0.5, 0.7, TEXT_COLOR, curved=0.2)
    draw_arrow(ax2, 2.5, 1.3, 3.0, 0.7, TEXT_COLOR, curved=0.2)
    draw_arrow(ax2, 0.45, 0.15, 3.05, 0.15, TEXT_COLOR, curved=-0.3)
    draw_self_loop(ax2, 3.5, 0.5, ACCENT_GREEN, "bottom")
    draw_self_loop(ax2, 0, 0.5, TEXT_COLOR, "bottom")

    ax2.text(3.2, 1.5, "Absorbing\nstate", color=ACCENT_GREEN,
             fontsize=9, fontweight="bold", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Figure 7: Crossover cartoon (diverse vs monoculture parents)
# =====================================================================

def make_crossover_cartoon(save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_bits = 10
    k = 3  # bits needed to reach optimum
    cell_w = 0.8
    cell_h = 0.5
    gap = 0.08

    def draw_bitstring(ax, x0, y0, bits, highlight_mask=None, label=None):
        """Draw a bitstring as colored cells."""
        for i, b in enumerate(bits):
            if highlight_mask is not None and highlight_mask[i]:
                color = ACCENT_GREEN if b == 1 else ACCENT_RED
            else:
                color = "#5a6a8a" if b == 1 else "#2a3a5a"
            rect = plt.Rectangle((x0 + i * (cell_w + gap), y0),
                                 cell_w, cell_h, facecolor=color,
                                 edgecolor=TEXT_COLOR, linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x0 + i * (cell_w + gap) + cell_w / 2, y0 + cell_h / 2,
                    str(b), ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        if label:
            ax.text(x0 - 0.3, y0 + cell_h / 2, label, ha="right", va="center",
                    fontsize=10, color=TEXT_COLOR, fontweight="bold")

    def draw_xover_arrow(ax, y_from, y_to, x_mid):
        arrow = FancyArrowPatch((x_mid, y_from), (x_mid, y_to),
                                arrowstyle="->,head_width=8,head_length=6",
                                color=ACCENT_BLUE, linewidth=2, zorder=10)
        ax.add_patch(arrow)

    # ---- Left panel: DIVERSE parents → successful crossover ----
    ax1.set_title("Diverse Parents → Crossover Works", fontsize=12,
                  fontweight="bold", color=ACCENT_GREEN, pad=10)

    # Optimum is all 1s (for simplicity)
    optimum = np.ones(n_bits, dtype=int)

    # Parent A: has the right bits on the left, wrong on the right
    parent_a = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
    # Parent B: has the right bits on the right, wrong on the left
    parent_b = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    # Crossover child: gets best of both
    child = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Highlight the "critical" bits each parent is missing
    missing_a = (parent_a != optimum)
    missing_b = (parent_b != optimum)

    total_w = n_bits * (cell_w + gap)
    x0 = 0.5

    draw_bitstring(ax1, x0, 3.0, parent_a, missing_a, "Parent A")
    draw_bitstring(ax1, x0, 2.0, parent_b, missing_b, "Parent B")
    draw_xover_arrow(ax1, 1.9, 1.1, x0 + total_w / 2)
    ax1.text(x0 + total_w / 2 + 0.5, 1.5, "Crossover", fontsize=10,
             color=ACCENT_BLUE, fontweight="bold", ha="left")
    draw_bitstring(ax1, x0, 0.3, child, np.zeros(n_bits, dtype=bool), "Child")
    ax1.text(x0 + total_w + 0.5, 0.55, "= Optimum!",
             fontsize=11, color=ACCENT_GREEN, fontweight="bold")

    ax1.set_xlim(-2, total_w + 3)
    ax1.set_ylim(-0.5, 4.2)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ---- Right panel: IDENTICAL parents → crossover useless ----
    ax2.set_title("Monoculture → Crossover Useless", fontsize=12,
                  fontweight="bold", color=ACCENT_RED, pad=10)

    clone = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
    missing_clone = (clone != optimum)
    child_clone = clone.copy()

    draw_bitstring(ax2, x0, 3.0, clone, missing_clone, "Parent A")
    draw_bitstring(ax2, x0, 2.0, clone, missing_clone, "Parent B")
    draw_xover_arrow(ax2, 1.9, 1.1, x0 + total_w / 2)
    ax2.text(x0 + total_w / 2 + 0.5, 1.5, "Crossover", fontsize=10,
             color=ACCENT_BLUE, fontweight="bold", ha="left")
    draw_bitstring(ax2, x0, 0.3, child_clone, missing_clone, "Child")
    ax2.text(x0 + total_w + 0.5, 0.55, "= Same gaps",
             fontsize=11, color=ACCENT_RED, fontweight="bold")

    ax2.set_xlim(-2, total_w + 3)
    ax2.set_ylim(-0.5, 4.2)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    figures = Path(__file__).parent.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("Generating figures for the deeper post...")
    print()

    make_phase_diagram(figures / "phase_diagram.png")
    make_time_series(figures / "time_series.png")
    make_takeover_curves(figures / "takeover_curves.png")
    make_mutation_phase_transition(figures / "mutation_phase_transition.png")
    make_self_adjusting_onemax(figures / "self_adjusting_onemax.png")
    make_markov_chain(figures / "markov_chain.png")
    make_crossover_cartoon(figures / "crossover_cartoon.png")

    print()
    print("Done! All figures saved to:", figures)
