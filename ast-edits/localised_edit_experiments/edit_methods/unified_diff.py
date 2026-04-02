"""Unified diff edit method."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from localised_edit_experiments.data_models import ApplyError, ParseError


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


class UnifiedDiffMethod:
    """Edit method using unified diff format."""

    def system_prompt(self) -> str:
        """Return system prompt for unified diff."""
        return (
            "You are a code editing assistant. When asked to edit code, "
            "output a unified diff in a single ```diff fenced block.\n\n"
            "Format requirements:\n"
            "- Use --- a/file and +++ b/file headers\n"
            "- Use @@ -START,COUNT +START,COUNT @@ hunk headers\n"
            "- Use 3 lines of context (lines starting with a space ' ')\n"
            "- Lines starting with '-' are removed, '+' are added\n"
            "- Context lines MUST match the original file exactly "
            "(including indentation)\n"
            "- Line numbers do not need to be exact; the context lines "
            "are used for matching\n\n"
            "Make ONLY the change requested. Do not include changes to "
            "any code beyond what the instruction asks for. "
            "Do not include any explanation outside the diff block."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with original code and instruction."""
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```python\n{original_code}\n```"
        )

    def parse(self, llm_output: str) -> list[DiffHunk]:
        """Extract and parse diff hunks from ```diff fenced block."""
        pattern = r"```diff\s*\n(.*?)```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if not match:
            raise ParseError("No ```diff fenced code block found in output")

        diff_text = match.group(1)
        hunks: list[DiffHunk] = []
        hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        current_hunk: DiffHunk | None = None
        for line in diff_text.splitlines():
            if line.startswith(("---", "+++")):
                continue

            header_match = hunk_header_re.match(line)
            if header_match:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                current_hunk = DiffHunk(
                    old_start=int(header_match.group(1)),
                    old_count=int(header_match.group(2) or "1"),
                    new_start=int(header_match.group(3)),
                    new_count=int(header_match.group(4) or "1"),
                )
                continue

            if current_hunk is not None and (
                line.startswith(("+", "-", " ")) or line == ""
            ):
                current_hunk.lines.append(line)

        if current_hunk is not None:
            hunks.append(current_hunk)

        if not hunks:
            raise ParseError("No diff hunks found in diff block")

        return hunks

    def apply(self, original_code: str, parsed_edit: list[DiffHunk]) -> str:
        """Apply diff hunks to original code."""
        original_lines = original_code.splitlines(keepends=True)
        # Ensure last line has newline for consistent processing
        if original_lines and not original_lines[-1].endswith("\n"):
            original_lines[-1] += "\n"

        offset = 0
        for hunk in parsed_edit:
            # 1-indexed to 0-indexed
            start = hunk.old_start - 1 + offset
            old_lines: list[str] = []
            new_lines: list[str] = []

            for line in hunk.lines:
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                elif line == "":
                    # Empty context line
                    old_lines.append("")
                    new_lines.append("")

            # Find the best matching position with tolerance
            best_pos = self._find_match(original_lines, old_lines, start)
            if best_pos is None:
                raise ApplyError(
                    f"Could not find matching context for hunk at line {hunk.old_start}"
                )

            # Add newlines to new_lines
            new_with_nl = [
                ln + "\n" if not ln.endswith("\n") else ln for ln in new_lines
            ]
            old_count = len(old_lines)
            original_lines[best_pos : best_pos + old_count] = new_with_nl
            offset += len(new_lines) - old_count

        result = "".join(original_lines)
        # Strip trailing newline if original didn't have one
        if not original_code.endswith("\n") and result.endswith("\n"):
            result = result.rstrip("\n")
        return result

    def _find_match(
        self,
        original_lines: list[str],
        old_lines: list[str],
        expected_pos: int,
    ) -> int | None:
        """Find position where old_lines match in original.

        First tries near expected_pos with increasing tolerance,
        then falls back to scanning the entire file.
        """
        if not old_lines:
            return expected_pos

        # Try near expected position first with expanding search
        tolerance = min(20, len(original_lines))
        for delta in range(tolerance + 1):
            for sign in (0, 1, -1):
                pos = expected_pos + delta * sign if sign else expected_pos
                if delta == 0 and sign != 0:
                    continue
                if pos < 0 or pos + len(old_lines) > len(original_lines):
                    continue
                if self._lines_match(original_lines, old_lines, pos):
                    return pos

        # Fall back to full-file scan
        for pos in range(len(original_lines) - len(old_lines) + 1):
            if self._lines_match(original_lines, old_lines, pos):
                return pos

        return None

    def _lines_match(
        self,
        original_lines: list[str],
        old_lines: list[str],
        pos: int,
    ) -> bool:
        """Check if old_lines match at position in original_lines."""
        for i, old_line in enumerate(old_lines):
            orig = original_lines[pos + i].rstrip("\n")
            if orig != old_line.rstrip("\n"):
                return False
        return True
