"""CLI entrypoint for localised code editing experiments."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np

from localised_edit_experiments.data_models import (
    EditMethodType,
    ExperimentResults,
)
from localised_edit_experiments.plotting import generate_all_plots, set_task_difficulty_map
from localised_edit_experiments.runner import ExperimentRunner
from localised_edit_experiments.tasks import get_tasks


_TASK_DIFFICULTY_MAP: dict[str, str] = {}


def _difficulty(task_id: str) -> str:
    """Classify a task by difficulty based on its ID prefix or lookup map."""
    if task_id in _TASK_DIFFICULTY_MAP:
        return _TASK_DIFFICULTY_MAP[task_id]
    for prefix in ("easy", "medium", "hard", "vhard", "extreme"):
        if task_id.startswith(prefix):
            return prefix
    return "super"


def _difficulty_breakdown(
    results: ExperimentResults,
    methods: list[str],
    models: list[str],
) -> list[str]:
    """Generate difficulty breakdown table lines."""
    diff_order = ["easy", "medium", "hard", "vhard", "extreme", "super"]
    diff_counts: dict[str, int] = {}
    seen_tasks: set[str] = set()
    for a in results.attempts:
        if a.task_id not in seen_tasks:
            seen_tasks.add(a.task_id)
            d = _difficulty(a.task_id)
            diff_counts[d] = diff_counts.get(d, 0) + 1
    active_diffs = [d for d in diff_order if diff_counts.get(d, 0) > 0]

    lines: list[str] = ["", "## Correctness by Difficulty", ""]
    header = "| Method | Model | " + " | ".join(active_diffs) + " |"
    sep = "|--------|-------|" + "|".join(
        "-" * (len(d) + 2) for d in active_diffs
    ) + "|"
    lines.extend([header, sep])

    for method in methods:
        for model in models:
            attempts = [
                a
                for a in results.attempts
                if a.method.value == method and a.model == model
            ]
            if not attempts:
                continue
            model_short = model.split("/")[-1] if "/" in model else model
            parts = []
            for d in active_diffs:
                d_attempts = [
                    a for a in attempts if _difficulty(a.task_id) == d
                ]
                correct = sum(1 for a in d_attempts if a.correct)
                parts.append(f"{correct}/{len(d_attempts)}")
            lines.append(
                f"| {method} | {model_short} | "
                + " | ".join(parts)
                + " |"
            )
    return lines


def _write_results_md(results: ExperimentResults, output_dir: Path) -> None:
    """Write results.md summary."""
    methods = list(dict.fromkeys(a.method.value for a in results.attempts))
    models = list(dict.fromkeys(a.model for a in results.attempts))
    task_ids = set(a.task_id for a in results.attempts)

    lines: list[str] = [
        "# Localised Code Editing Experiment Results",
        "",
        f"**Started:** {results.started_at:%Y-%m-%d %H:%M:%S}",
        f"**Finished:** {results.finished_at:%Y-%m-%d %H:%M:%S}"
        if results.finished_at
        else "",
        f"**Tasks:** {len(task_ids)}",
        f"**Models:** {', '.join(models)}",
        f"**Methods:** {', '.join(methods)}",
        f"**Total attempts:** {len(results.attempts)}",
        "",
        "## Summary Table",
        "",
        "| Method | Model | Apply% | Correct% | "
        "Avg In Tok | Avg Out Tok | Avg Latency | "
        "Avg Minimality | Avg CD Score |",
        "|--------|-------|--------|----------|"
        "-----------|-------------|-------------|"
        "----------------|--------------|",
    ]

    for method in methods:
        for model in models:
            attempts = [
                a
                for a in results.attempts
                if a.method.value == method and a.model == model
            ]
            if not attempts:
                continue
            n = len(attempts)
            apply_rate = sum(a.apply_success for a in attempts) / n
            correct_rate = sum(a.correct for a in attempts) / n
            mean_in = np.mean([a.input_tokens for a in attempts])
            mean_out = np.mean([a.output_tokens for a in attempts])
            mean_latency = np.mean([a.latency_s for a in attempts])
            correct_attempts = [a for a in attempts if a.correct]
            mean_min = (
                np.mean([a.edit_minimality for a in correct_attempts])
                if correct_attempts
                else 0.0
            )
            mean_cd = (
                np.mean(
                    [a.collateral_damage_score for a in correct_attempts]
                )
                if correct_attempts
                else 0.0
            )
            model_short = model.split("/")[-1] if "/" in model else model
            lines.append(
                f"| {method} | {model_short} | "
                f"{apply_rate:.1%} | {correct_rate:.1%} | "
                f"{mean_in:.0f} | {mean_out:.0f} | "
                f"{mean_latency:.1f}s | "
                f"{mean_min:.2f} | {mean_cd:.2f} |"
            )

    lines.extend(_difficulty_breakdown(results, methods, models))

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Success Rates](success_rates.png)",
            "![Token Usage](token_usage.png)",
            "![Edit Minimality](edit_minimality.png)",
            "![Latency](latency.png)",
            "![Difficulty by Model](difficulty_by_model.png)",
            "",
            "## Example Outputs",
            "",
        ]
    )

    # Add best and worst examples per method
    for method in methods:
        method_attempts = [a for a in results.attempts if a.method.value == method]
        correct = [a for a in method_attempts if a.correct]
        failed = [a for a in method_attempts if not a.apply_success]

        lines.append(f"### {method}")
        lines.append("")

        if correct:
            best = min(correct, key=lambda a: a.latency_s)
            lines.append(
                f"**Best (task={best.task_id}, latency={best.latency_s:.1f}s):**"
            )
            lines.append("```")
            lines.append(best.raw_llm_output[:500])
            lines.append("```")
            lines.append("")

        if failed:
            worst = failed[0]
            lines.append(f"**Failed (task={worst.task_id}):**")
            lines.append(f"Error: {worst.error_message}")
            lines.append("")

    (output_dir / "results.md").write_text("\n".join(lines))


def _serialize_results(results: ExperimentResults) -> str:
    """Serialize results to JSON."""
    data = asdict(results)
    # Convert datetime and enum to string
    for attempt in data["attempts"]:
        attempt["method"] = (
            attempt["method"].value
            if hasattr(attempt["method"], "value")
            else attempt["method"]
        )
    data["started_at"] = data["started_at"].isoformat()
    if data["finished_at"]:
        data["finished_at"] = data["finished_at"].isoformat()
    return json.dumps(data, indent=2, default=str)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results",
    help="Output directory for results.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=["claude-haiku-4-5-20251001"],
    help="LLM models to test (repeatable).",
)
@click.option(
    "--method",
    "methods",
    multiple=True,
    default=[m.value for m in EditMethodType],
    type=click.Choice([m.value for m in EditMethodType]),
    help="Edit methods to test (repeatable).",
)
@click.option(
    "--sleep",
    "sleep_between",
    type=float,
    default=1.0,
    help="Seconds to sleep between API calls.",
)
def main(
    output_dir: str,
    models: tuple[str, ...],
    methods: tuple[str, ...],
    sleep_between: float,
) -> None:
    """Run localised code editing experiments."""
    out = Path(output_dir)
    method_types = [EditMethodType(m) for m in methods]

    tasks = get_tasks()
    click.echo(f"Using built-in tasks ({len(tasks)} tasks)")

    # Populate difficulty map for reporting and plotting
    diff_map = {t.task_id: t.difficulty.value for t in tasks}
    _TASK_DIFFICULTY_MAP.update(diff_map)
    set_task_difficulty_map(diff_map)

    runner = ExperimentRunner(
        models=list(models),
        methods=method_types,
        output_dir=out,
        sleep_between=sleep_between,
    )

    results = asyncio.run(runner.run_all(tasks=tasks))

    # Save raw results
    (out / "results.json").write_text(_serialize_results(results))

    # Generate plots
    generate_all_plots(results, out)

    # Write results markdown
    _write_results_md(results, out)

    click.echo(f"Results written to {out}")


if __name__ == "__main__":
    main()
