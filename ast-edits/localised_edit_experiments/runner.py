"""Experiment runner for localised code editing experiments."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from litellm import acompletion

from localised_edit_experiments.data_models import (
    EditAttempt,
    EditMethodType,
    ExperimentResults,
    ParseError,
)
from localised_edit_experiments.edit_methods import (
    AstEditMethod,
    HashlineJsonOpsMethod,
    HashlineSearchReplaceMethod,
    HashlineUnifiedDiffMethod,
    SearchReplaceMethod,
    UnifiedDiffMethod,
    WholeFileMethod,
)
from localised_edit_experiments.evaluation import (
    compute_collateral_damage,
    compute_correctness,
    compute_edit_minimality,
)
from localised_edit_experiments.tasks import get_tasks


if TYPE_CHECKING:
    from pathlib import Path

    from localised_edit_experiments.data_models import EditTask
    from localised_edit_experiments.edit_methods.base import EditMethod


load_dotenv()

logger = logging.getLogger(__name__)

METHOD_MAP: dict[EditMethodType, type[EditMethod]] = {
    EditMethodType.WHOLE_FILE: WholeFileMethod,
    EditMethodType.UNIFIED_DIFF: UnifiedDiffMethod,
    EditMethodType.SEARCH_REPLACE: SearchReplaceMethod,
    EditMethodType.AST_EDIT: AstEditMethod,
    EditMethodType.HASHLINE_JSON_OPS: HashlineJsonOpsMethod,
    EditMethodType.HASHLINE_UNIFIED_DIFF: HashlineUnifiedDiffMethod,
    EditMethodType.HASHLINE_SEARCH_REPLACE: HashlineSearchReplaceMethod,
}


class ExperimentRunner:
    """Orchestrates localised code editing experiments."""

    def __init__(
        self,
        models: list[str],
        methods: list[EditMethodType],
        output_dir: Path,
        sleep_between: float = 1.0,
    ) -> None:
        self.models = models
        self.methods = methods
        self.output_dir = output_dir
        self.sleep_between = sleep_between
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure file and console logging."""
        log_file = self.output_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    async def run_single(
        self,
        model: str,
        method_type: EditMethodType,
        task: EditTask,
    ) -> EditAttempt:
        """Run a single LLM edit attempt."""
        method = METHOD_MAP[method_type]()
        model_short = model.split("/")[-1] if "/" in model else model

        max_retries = 5
        retry_wait = 60  # seconds
        start_time = time.monotonic()

        # O-series and GPT-5+ models don't support temperature
        is_no_temp = any(
            f"o{n}" in model_short for n in ("1", "3", "4")
        ) or model_short.startswith("gpt-5")
        extra_kwargs: dict[str, float] = {}
        if not is_no_temp:
            extra_kwargs["temperature"] = 0.0

        for attempt_num in range(max_retries + 1):
            try:
                response = await acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": method.system_prompt()},
                        {
                            "role": "user",
                            "content": method.user_prompt(
                                task.original_code, task.description
                            ),
                        },
                    ],
                    **extra_kwargs,
                )
                break  # success
            except Exception as e:
                is_retryable = (
                    "rate" in str(e).lower()
                    or "429" in str(e)
                    or "RateLimitError" in type(e).__name__
                    or "503" in str(e)
                    or "500" in str(e)
                    or "ServiceUnavailable" in type(e).__name__
                    or "InternalServerError" in type(e).__name__
                )
                if is_retryable and attempt_num < max_retries:
                    logger.info(
                        "model=%s method=%s task=%s | RETRYABLE ERROR: %s (attempt %d/%d) — "
                        "waiting %ds before retry...",
                        model_short,
                        method_type.value,
                        task.task_id,
                        str(e)[:80],
                        attempt_num + 1,
                        max_retries,
                        retry_wait,
                    )
                    await asyncio.sleep(retry_wait)
                    continue

                latency = time.monotonic() - start_time
                logger.info(
                    "model=%s method=%s task=%s | ERROR %s latency=%.1fs",
                    model_short,
                    method_type.value,
                    task.task_id,
                    str(e)[:80],
                    latency,
                )
                return EditAttempt(
                    task_id=task.task_id,
                    method=method_type,
                    model=model,
                    raw_llm_output="",
                    applied_code=None,
                    apply_success=False,
                    correct=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=latency,
                    edit_minimality=0.0,
                    error_message=str(e),
                )

        latency = time.monotonic() - start_time
        raw_output = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Parse and apply
        applied_code = None
        apply_success = False
        correct = False
        edit_minimality = 0.0
        collateral_score = 1.0
        error_message = None

        try:
            parsed = method.parse(raw_output)
            applied_code = method.apply(task.original_code, parsed)
            apply_success = True
            correct = compute_correctness(
                applied_code, task.test_code, task.original_code
            )
            if correct:
                edit_minimality = compute_edit_minimality(
                    task.original_code, applied_code, task.expected_code
                )
                collateral_score = compute_collateral_damage(
                    task.original_code, applied_code, task.expected_code
                )
        except (ParseError, Exception) as e:
            error_message = str(e)

        status = "SUCCESS" if apply_success else "FAIL"
        logger.info(
            "model=%s method=%s task=%s | %s correct=%s tokens=%d latency=%.1fs",
            model_short,
            method_type.value,
            task.task_id,
            status,
            correct,
            input_tokens + output_tokens,
            latency,
        )

        return EditAttempt(
            task_id=task.task_id,
            method=method_type,
            model=model,
            raw_llm_output=raw_output,
            applied_code=applied_code,
            apply_success=apply_success,
            correct=correct,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            edit_minimality=edit_minimality,
            collateral_damage_score=collateral_score,
            error_message=error_message,
        )

    async def run_all(
        self, tasks: list[EditTask] | None = None,
    ) -> ExperimentResults:
        """Run all experiment combinations sequentially."""
        if tasks is None:
            tasks = get_tasks()
        results = ExperimentResults(started_at=datetime.now(tz=UTC))

        total = len(self.models) * len(self.methods) * len(tasks)
        logger.info(
            "Starting experiment: %d models x %d methods x %d tasks = %d calls",
            len(self.models),
            len(self.methods),
            len(tasks),
            total,
        )

        count = 0
        for model in self.models:
            for method_type in self.methods:
                for task in tasks:
                    attempt = await self.run_single(model, method_type, task)
                    results.attempts.append(attempt)
                    count += 1
                    if count < total:
                        await asyncio.sleep(self.sleep_between)

        results.finished_at = datetime.now(tz=UTC)
        logger.info(
            "Experiment complete: %d attempts in %.1f minutes",
            len(results.attempts),
            (results.finished_at - results.started_at).total_seconds() / 60,
        )
        return results
