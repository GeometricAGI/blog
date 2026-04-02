"""Evaluation metrics for localised code editing experiments."""

from __future__ import annotations

import difflib
import logging
import re
import signal


logger = logging.getLogger(__name__)

_EXEC_TIMEOUT_SECONDS = 10


def normalize_code(code: str) -> str:
    """Normalize code for comparison.

    Strips trailing whitespace per line, collapses multiple blank lines
    into one, and strips leading/trailing blank lines.
    """
    lines = code.splitlines()
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in lines]
    # Collapse multiple consecutive blank lines into one
    normalized: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        normalized.append(line)
        prev_blank = is_blank
    # Strip leading and trailing blank lines
    while normalized and normalized[0] == "":
        normalized.pop(0)
    while normalized and normalized[-1] == "":
        normalized.pop()
    return "\n".join(normalized)


def _timeout_handler(signum: int, frame: object) -> None:
    raise TimeoutError("Code execution timed out")


def compute_correctness(
    actual_code: str, test_code: str, original_code: str = ""
) -> bool:
    """Check if actual code passes the test assertions.

    Execs the LLM-produced code, then runs the test_code assertions
    in the same namespace. Returns True if all assertions pass.
    """
    if not test_code.strip():
        return False

    namespace: dict[str, object] = {}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    passed = False
    try:
        signal.alarm(_EXEC_TIMEOUT_SECONDS)
        exec(compile(actual_code, "<edit_output>", "exec"), namespace)
        namespace["_code_under_test"] = actual_code
        namespace["_original_code"] = original_code
        exec(compile(test_code, "<test_code>", "exec"), namespace)
        passed = True
    except Exception as exc:
        logger.debug("Correctness check failed: %s", exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return passed


def compute_edit_minimality(original: str, actual: str, expected: str) -> float:
    """Compute edit minimality as ratio of expected diff size to actual diff size.

    Returns a value in [0, 1] where 1.0 means perfectly minimal edits.
    """
    expected_diff = _diff_size(original, expected)
    actual_diff = _diff_size(original, actual)

    if actual_diff == 0:
        return 1.0 if expected_diff == 0 else 0.0
    return min(expected_diff / actual_diff, 1.0)


def compute_collateral_damage(
    original: str, actual: str, expected: str, margin: int = 20
) -> float:
    """Measure how much the LLM changed code outside the expected edit regions.

    Returns a score in [0, 1] where 1.0 means no collateral damage
    (all changes fall within or near the expected edit regions) and 0.0
    means most changes are to untouched code.

    Args:
        original: The original code before editing.
        actual: The LLM's output code.
        expected: The reference correct code.
        margin: Number of lines of tolerance around each edit region.
    """
    orig_lines = normalize_code(original).splitlines()
    actual_lines = normalize_code(actual).splitlines()
    expected_lines = normalize_code(expected).splitlines()

    # Find which lines changed in original→expected (the edit regions)
    expected_changed = _changed_line_numbers(orig_lines, expected_lines)
    if not expected_changed:
        # No expected changes — any actual change is collateral
        actual_changed = _changed_line_numbers(orig_lines, actual_lines)
        return 1.0 if not actual_changed else 0.0

    # Expand edit regions by margin
    max_line = max(len(orig_lines), len(actual_lines), len(expected_lines))
    allowed_region: set[int] = set()
    for line_no in expected_changed:
        for offset in range(-margin, margin + 1):
            clamped = max(0, min(line_no + offset, max_line))
            allowed_region.add(clamped)

    # Find which lines changed in original→actual
    actual_changed = _changed_line_numbers(orig_lines, actual_lines)
    if not actual_changed:
        return 1.0

    # Count how many actual changes fall outside the allowed region
    collateral = actual_changed - allowed_region
    total_actual = len(actual_changed)

    return 1.0 - (len(collateral) / total_actual) if total_actual > 0 else 1.0


def _changed_line_numbers(a_lines: list[str], b_lines: list[str]) -> set[int]:
    """Return 0-indexed line numbers that differ between a and b.

    Uses SequenceMatcher to align the sequences properly, so insertions
    and deletions are tracked by their position in the original (a) side.
    """
    matcher = difflib.SequenceMatcher(None, a_lines, b_lines)
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            for i in range(i1, i2):
                changed.add(i)
    return changed


def _diff_size(a: str, b: str) -> int:
    """Count the number of changed lines between two code strings."""
    a_lines = normalize_code(a).splitlines()
    b_lines = normalize_code(b).splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, lineterm="")
    # Count only +/- lines (not headers or context)
    count = 0
    for line in diff:
        if re.match(r"^[+-](?![+-]{2})", line):
            count += 1
    return count
