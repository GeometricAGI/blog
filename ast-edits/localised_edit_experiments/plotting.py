"""Plotting functions for localised code editing experiment results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from localised_edit_experiments.tasks import get_tasks


if TYPE_CHECKING:
    from pathlib import Path

    from localised_edit_experiments.data_models import (
        EditAttempt,
        ExperimentResults,
    )


matplotlib.use("Agg")


_DIFFICULTY_OVERRIDE: dict[str, str] = {}


def set_task_difficulty_map(mapping: dict[str, str]) -> None:
    """Set a custom task difficulty map (e.g. for CanItEdit tasks)."""
    _DIFFICULTY_OVERRIDE.clear()
    _DIFFICULTY_OVERRIDE.update(mapping)


def _get_task_difficulty_map() -> dict[str, str]:
    """Get mapping of task_id to difficulty value."""
    if _DIFFICULTY_OVERRIDE:
        return dict(_DIFFICULTY_OVERRIDE)
    return {t.task_id: t.difficulty.value for t in get_tasks()}


def _group_attempts(
    results: ExperimentResults,
) -> dict[tuple[str, str], list[EditAttempt]]:
    """Group attempts by (method, model)."""
    groups: dict[tuple[str, str], list[EditAttempt]] = {}
    for attempt in results.attempts:
        key = (attempt.method.value, attempt.model)
        groups.setdefault(key, []).append(attempt)
    return groups


def _get_methods_and_models(
    results: ExperimentResults,
) -> tuple[list[str], list[str]]:
    """Extract unique methods and models from results."""
    methods = list(dict.fromkeys(a.method.value for a in results.attempts))
    models = list(dict.fromkeys(a.model for a in results.attempts))
    return methods, models


def _short_model(model: str) -> str:
    """Shorten model name for display."""
    return model.split("/")[-1] if "/" in model else model


def plot_success_rates(results: ExperimentResults, output_dir: Path) -> None:
    """Plot grouped bar chart of apply success and correctness rates."""
    groups = _group_attempts(results)
    methods, models = _get_methods_and_models(results)

    x = np.arange(len(methods))
    width = 0.35 / len(models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, model in enumerate(models):
        apply_rates = []
        correct_rates = []
        for method in methods:
            attempts = groups.get((method, model), [])
            if attempts:
                apply_rates.append(
                    sum(a.apply_success for a in attempts) / len(attempts)
                )
                correct_rates.append(sum(a.correct for a in attempts) / len(attempts))
            else:
                apply_rates.append(0.0)
                correct_rates.append(0.0)

        offset = (i - len(models) / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            apply_rates,
            width,
            label=_short_model(model),
        )
        ax2.bar(
            x + offset,
            correct_rates,
            width,
            label=_short_model(model),
        )

    for ax, title in [
        (ax1, "Apply Success Rate"),
        (ax2, "Correctness Rate"),
    ]:
        ax.set_xlabel("Edit Method")
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=150)
    plt.close(fig)


def plot_token_usage(results: ExperimentResults, output_dir: Path) -> None:
    """Plot grouped bar chart of mean token usage."""
    groups = _group_attempts(results)
    methods, models = _get_methods_and_models(results)

    x = np.arange(len(methods))
    width = 0.35 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        output_tokens = []
        for method in methods:
            attempts = groups.get((method, model), [])
            if attempts:
                output_tokens.append(np.mean([a.output_tokens for a in attempts]))
            else:
                output_tokens.append(0)

        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            output_tokens,
            width,
            label=_short_model(model),
        )

    ax.set_xlabel("Edit Method")
    ax.set_ylabel("Mean Output Tokens")
    ax.set_title("Output Token Usage by Method and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "token_usage.png", dpi=150)
    plt.close(fig)


def plot_edit_minimality(results: ExperimentResults, output_dir: Path) -> None:
    """Plot box plot of edit minimality per method (correct attempts only)."""
    methods, _models = _get_methods_and_models(results)

    data_by_method: dict[str, list[float]] = {m: [] for m in methods}
    for attempt in results.attempts:
        if attempt.correct:
            data_by_method[attempt.method.value].append(attempt.edit_minimality)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [data_by_method[m] for m in methods]
    # Handle empty data
    data = [d if d else [0.0] for d in data]
    ax.boxplot(data, tick_labels=methods)
    ax.set_xlabel("Edit Method")
    ax.set_ylabel("Edit Minimality")
    ax.set_title("Edit Minimality (Correct Attempts Only)")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(output_dir / "edit_minimality.png", dpi=150)
    plt.close(fig)


def plot_latency(results: ExperimentResults, output_dir: Path) -> None:
    """Plot bar chart of mean latency per method/model."""
    groups = _group_attempts(results)
    methods, models = _get_methods_and_models(results)

    x = np.arange(len(methods))
    width = 0.35 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        latencies = []
        for method in methods:
            attempts = groups.get((method, model), [])
            if attempts:
                latencies.append(np.mean([a.latency_s for a in attempts]))
            else:
                latencies.append(0.0)

        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            latencies,
            width,
            label=_short_model(model),
        )

    ax.set_xlabel("Edit Method")
    ax.set_ylabel("Mean Latency (s)")
    ax.set_title("Latency by Method and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "latency.png", dpi=150)
    plt.close(fig)


def plot_success_by_difficulty(results: ExperimentResults, output_dir: Path) -> None:
    """Plot heatmap of correctness by difficulty and method."""
    methods, _models = _get_methods_and_models(results)
    difficulties = ["easy", "medium", "hard"]

    task_difficulty = _get_task_difficulty_map()

    # Build matrix: rows=difficulties, cols=methods
    matrix = np.zeros((len(difficulties), len(methods)))
    counts = np.zeros((len(difficulties), len(methods)))

    for attempt in results.attempts:
        diff = task_difficulty.get(attempt.task_id, "unknown")
        if diff in difficulties:
            row = difficulties.index(diff)
            col = methods.index(attempt.method.value)
            counts[row, col] += 1
            if attempt.correct:
                matrix[row, col] += 1

    # Convert to percentages
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(counts > 0, matrix / counts * 100, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pct, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(difficulties)))
    ax.set_yticklabels(difficulties)
    ax.set_title("Correctness % by Difficulty and Method")

    # Add text annotations
    for i in range(len(difficulties)):
        for j in range(len(methods)):
            ax.text(
                j,
                i,
                f"{pct[i, j]:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.colorbar(im, ax=ax, label="Correctness %")
    plt.tight_layout()
    fig.savefig(output_dir / "success_by_difficulty.png", dpi=150)
    plt.close(fig)


def plot_difficulty_by_model(results: ExperimentResults, output_dir: Path) -> None:
    """Plot correctness by difficulty as grouped bar charts, one per model.

    Each subplot shows one model. Within each subplot, difficulties are
    on the x-axis and methods are grouped bars with consistent colours.
    """
    methods, models = _get_methods_and_models(results)
    difficulties = ["easy", "medium", "hard"]
    task_difficulty = _get_task_difficulty_map()

    cmap = plt.get_cmap("tab10")
    method_colors = {m: cmap(i) for i, m in enumerate(methods)}

    n_models = len(models)
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(difficulties))
    bar_width = 0.8 / n_methods

    for ax, model in zip(axes, models):
        for j, method in enumerate(methods):
            pcts = []
            for diff in difficulties:
                attempts = [
                    a
                    for a in results.attempts
                    if a.model == model
                    and a.method.value == method
                    and task_difficulty.get(a.task_id) == diff
                ]
                total = len(attempts)
                correct = sum(a.correct for a in attempts)
                pcts.append(correct / total * 100 if total > 0 else 0)

            offset = (j - n_methods / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                pcts,
                bar_width,
                color=method_colors[method],
                label=method,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.set_xlabel("Difficulty")
        ax.set_title(_short_model(model))
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Correctness %")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    fig.suptitle("Correctness by Difficulty and Method", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "difficulty_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results: ExperimentResults, output_dir: Path) -> None:
    """Generate all experiment plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_success_rates(results, output_dir)
    plot_token_usage(results, output_dir)
    plot_edit_minimality(results, output_dir)
    plot_latency(results, output_dir)
    plot_difficulty_by_model(results, output_dir)
