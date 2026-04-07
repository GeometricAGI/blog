"""Diversity comparison animations for the Rastrigin function.

Generates three GIFs showing the effect of diversity on evolutionary search:
1. Too little diversity — population collapses to a local minimum
2. Too much diversity — population wanders randomly, never converges
3. Balanced diversity — population finds the global minimum
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


def rastrigin(x, y, A=10):
    """Rastrigin function — canonical multimodal benchmark."""
    return (A * 2
            + (x**2 - A * np.cos(2 * np.pi * x))
            + (y**2 - A * np.cos(2 * np.pi * y)))


def _build_history(population, fitness, frames, bound, mode):
    """Run the EA and record population history.

    mode: 'low', 'high', or 'balanced'
    """
    pop_size = len(population)
    history = [(population.copy(), fitness.copy())]

    for gen in range(frames - 1):
        gen_frac = gen / frames

        if mode == "low":
            # Very aggressive selection (tournament size 10) + tiny mutation
            new_pop = np.empty_like(population)
            for j in range(pop_size):
                k = min(10, pop_size)
                candidates = np.random.choice(pop_size, k, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                new_pop[j] = population[winner]
            sigma = 0.15
            mutation = np.random.normal(0, sigma, (pop_size, 2))
            new_pop = np.clip(new_pop + mutation, -bound, bound)

        elif mode == "high":
            # Weak selection (tournament size 2) + huge mutation
            new_pop = np.empty_like(population)
            for j in range(pop_size):
                candidates = np.random.choice(pop_size, 2, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                new_pop[j] = population[winner]
            sigma = 3.5
            mutation = np.random.normal(0, sigma, (pop_size, 2))
            new_pop = np.clip(new_pop + mutation, -bound, bound)

        else:  # balanced
            # Moderate-strong selection (tournament size 5) + adaptive mutation
            new_pop = np.empty_like(population)
            for j in range(pop_size):
                candidates = np.random.choice(pop_size, 5, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                new_pop[j] = population[winner]
            sigma = 1.2 * (1 - 0.8 * gen_frac)
            mutation = np.random.normal(0, sigma, (pop_size, 2))
            new_pop = np.clip(new_pop + mutation, -bound, bound)

        # Elitism: keep the best
        best_idx = np.argmin(fitness)
        new_fitness = rastrigin(new_pop[:, 0], new_pop[:, 1])
        worst_new_idx = np.argmax(new_fitness)
        new_pop[worst_new_idx] = population[best_idx]

        population = new_pop
        fitness = rastrigin(population[:, 0], population[:, 1])
        history.append((population.copy(), fitness.copy()))

    return history


def _render_animation(history, frames, bound, subtitle, outcome_label,
                      outcome_color, save_path, res=200, figsize=(6, 6),
                      fps=4):
    """Render and save a single animation GIF."""
    # Build heatmap grid
    x_lin = np.linspace(-bound, bound, res)
    y_lin = np.linspace(-bound, bound, res)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = rastrigin(X, Y)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.pcolormesh(X, Y, Z, cmap="inferno", shading="auto", alpha=0.85)

    # Global minimum marker
    ax.plot(0, 0, "*", color="#4cc9f0", markersize=16, zorder=10,
            markeredgecolor="white", markeredgewidth=0.8)

    scatter = ax.scatter(
        [], [], c=[], cmap="RdYlGn_r", s=50,
        edgecolors="white", linewidths=0.6, zorder=5,
        vmin=0, vmax=Z.max() * 0.5,
    )

    title = ax.set_title("", color="white", fontsize=13, fontweight="bold")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")

    best_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, color="#4cc9f0",
        fontsize=9, fontweight="bold", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e",
                  edgecolor="#4a4a6a", alpha=0.9),
    )

    outcome_text = ax.text(
        0.5, 0.03, "", transform=ax.transAxes, color=outcome_color,
        fontsize=11, fontweight="bold", verticalalignment="bottom",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                  edgecolor=outcome_color, alpha=0.9),
    )

    def update(frame):
        pop, fit = history[frame]
        scatter.set_offsets(pop)
        scatter.set_array(fit)
        best_val = fit.min()
        best_pos = pop[np.argmin(fit)]
        title.set_text(f"{subtitle} — Gen {frame + 1}/{frames}")
        best_text.set_text(
            f"Best: f({best_pos[0]:.2f}, {best_pos[1]:.2f}) = {best_val:.3f}"
        )
        # Show outcome label in last 25% of frames
        if frame >= frames * 0.75:
            outcome_text.set_text(outcome_label)
        else:
            outcome_text.set_text("")
        return scatter, title, best_text, outcome_text

    anim = FuncAnimation(fig, update, frames=frames, interval=250, blit=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def run_low(save_path=None, frames=60, pop_size=30):
    """Too little diversity — stuck in local minimum."""
    np.random.seed(42)
    bound = 5.12
    # Start population away from global min, near a local minimum
    population = np.random.uniform(2.0, 4.5, (pop_size, 2))
    fitness = rastrigin(population[:, 0], population[:, 1])
    history = _build_history(population, fitness, frames, bound, "low")
    _render_animation(
        history, frames, bound,
        subtitle="Low Diversity",
        outcome_label="Stuck in local optimum",
        outcome_color="#f72585",
        save_path=save_path,
        res=150, figsize=(6, 6), fps=4,
    )


def run_high(save_path=None, frames=60, pop_size=30):
    """Too much diversity — never converges."""
    np.random.seed(42)
    bound = 5.12
    population = np.random.uniform(-bound, bound, (pop_size, 2))
    fitness = rastrigin(population[:, 0], population[:, 1])
    history = _build_history(population, fitness, frames, bound, "high")
    _render_animation(
        history, frames, bound,
        subtitle="High Diversity",
        outcome_label="No convergence",
        outcome_color="#f8961e",
        save_path=save_path,
        # Smaller resolution + fewer frames to keep file size down
        res=150, figsize=(6, 6), fps=4,
    )


def run_balanced(save_path=None, frames=80, pop_size=30):
    """Just right — finds the global minimum."""
    np.random.seed(42)
    bound = 5.12
    population = np.random.uniform(-bound, bound, (pop_size, 2))
    fitness = rastrigin(population[:, 0], population[:, 1])
    history = _build_history(population, fitness, frames, bound, "balanced")
    _render_animation(
        history, frames, bound,
        subtitle="Balanced Diversity",
        outcome_label="Global optimum found!",
        outcome_color="#4ade80",
        save_path=save_path,
        res=200, figsize=(6, 6), fps=4,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Show interactively instead of saving")
    args = parser.parse_args()

    figures = Path(__file__).parent.parent / "figures"

    if args.show:
        run_low()
        run_high()
        run_balanced()
    else:
        run_low(save_path=str(figures / "diversity_low.gif"))
        run_high(save_path=str(figures / "diversity_high.gif"))
        run_balanced(save_path=str(figures / "diversity_balanced.gif"))
