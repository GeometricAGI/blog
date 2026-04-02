"""Tag-anchored hashline unified diff method.

Instead of requiring the LLM to reproduce context/removed lines perfectly,
hunk headers use tag ranges to anchor the edit location. Only added lines
('+') need to be written. Removed ('-') and context (' ') lines are optional
hints that are ignored during apply — the tag range is authoritative.

Inspired by the approach described in "The Harness Problem" (Can Bölük, 2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from localised_edit_experiments.data_models import ApplyError, ParseError
from localised_edit_experiments.edit_methods.hashline import (
    _build_hash_map,
    tag_lines,
)


@dataclass
class TaggedDiffHunk:
    """A diff hunk anchored by hashline tags."""

    start_tag: str
    end_tag: str
    added_lines: list[str] = field(default_factory=list)


class HashlineUnifiedDiffMethod:
    """Unified diff using tag-anchored hunk headers."""

    def system_prompt(self) -> str:
        """Return system prompt for tag-anchored unified diff."""
        return (
            "You are a code editing assistant. The original code is shown "
            "with each line tagged as `LINENO:HASH|content`, for example:\n\n"
            "```\n"
            "1:a3|def hello():\n"
            '2:f1|    return "world"\n'
            "3:0e|\n"
            "```\n\n"
            "When asked to edit code, output a diff in a single ```diff "
            "fenced block using tag-anchored hunk headers.\n\n"
            "Format:\n"
            "- Use `@@ START_TAG..END_TAG @@` hunk headers to specify which "
            "lines to replace (inclusive range)\n"
            "- Lines starting with '+' are the new replacement content\n"
            "- Lines starting with '-' or ' ' are optional hints (ignored "
            "during apply — the tag range is authoritative)\n"
            "- Tags must be copied EXACTLY from the tagged input\n\n"
            "To replace a range of lines with new content:\n"
            "```\n"
            "@@ 2:f1..2:f1 @@\n"
            '+    return "hello"\n'
            "```\n\n"
            "To delete lines (no '+' lines):\n"
            "```\n"
            "@@ 2:f1..3:0e @@\n"
            "```\n\n"
            "To insert new lines AFTER a line (without removing it), use "
            "the INSERT keyword:\n"
            "```\n"
            "@@ INSERT 2:f1 @@\n"
            "+    new_line_here()\n"
            "```\n\n"
            "Rules:\n"
            "- Tags must be copied EXACTLY from the input (e.g., "
            '"2:f1" not "2:f2" or just "2")\n'
            "- You do NOT need to reproduce old code — just specify the "
            "tag range and the new content\n"
            "- Only '+' lines in the replacement are used; '-' and ' ' "
            "lines are optional commentary\n"
            "- You may use multiple hunks for multiple changes\n"
            "- Make ONLY the change requested\n\n"
            "Example — changing a return value on line 2:\n"
            "```diff\n"
            "@@ 2:f1..2:f1 @@\n"
            '+    return "hello"\n'
            "```\n\n"
            "Do not include any explanation outside the diff block."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with hash-tagged code."""
        tagged = tag_lines(original_code)
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```\n{tagged}\n```"
        )

    def parse(self, llm_output: str) -> list[TaggedDiffHunk]:
        """Extract tag-anchored hunks from diff block."""
        pattern = r"```diff\s*\n(.*?)```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if not match:
            raise ParseError("No ```diff fenced code block found in output")

        diff_text = match.group(1)
        hunks: list[TaggedDiffHunk] = []

        # Match range hunks: @@ START_TAG..END_TAG @@
        range_header_re = re.compile(
            r"^@@\s+(\d+:[a-f0-9]{2})\.\.(\d+:[a-f0-9]{2})\s+@@"
        )
        # Match insert hunks: @@ INSERT TAG @@
        insert_header_re = re.compile(
            r"^@@\s+INSERT\s+(\d+:[a-f0-9]{2})\s+@@"
        )

        current_hunk: TaggedDiffHunk | None = None

        for line in diff_text.splitlines():
            # Skip --- / +++ headers if present
            if line.startswith(("---", "+++")):
                continue

            range_match = range_header_re.match(line)
            insert_match = insert_header_re.match(line)

            if range_match:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                current_hunk = TaggedDiffHunk(
                    start_tag=range_match.group(1),
                    end_tag=range_match.group(2),
                )
                continue

            if insert_match:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                # For INSERT, we use a sentinel: start_tag == end_tag with
                # a special marker. We'll handle this in apply.
                current_hunk = TaggedDiffHunk(
                    start_tag="INSERT:" + insert_match.group(1),
                    end_tag=insert_match.group(1),
                )
                continue

            if current_hunk is not None:
                if line.startswith("+"):
                    current_hunk.added_lines.append(line[1:])
                # '-' and ' ' lines are ignored (optional hints)

        if current_hunk is not None:
            hunks.append(current_hunk)

        if not hunks:
            raise ParseError("No tag-anchored hunks found in diff block")

        return hunks

    def apply(
        self, original_code: str, parsed_edit: list[TaggedDiffHunk]
    ) -> str:
        """Apply tag-anchored hunks to original code."""
        lines = original_code.splitlines()
        hash_map = _build_hash_map(original_code)

        resolved: list[tuple[int, int, bool, list[str]]] = []
        for hunk in parsed_edit:
            is_insert = hunk.start_tag.startswith("INSERT:")
            if is_insert:
                actual_tag = hunk.start_tag[len("INSERT:"):]
                idx = self._resolve_tag(actual_tag, hash_map)
                resolved.append((idx, idx, True, hunk.added_lines))
            else:
                start = self._resolve_tag(hunk.start_tag, hash_map)
                end = self._resolve_tag(hunk.end_tag, hash_map)
                resolved.append((start, end, False, hunk.added_lines))

        # Process in reverse to preserve positions
        resolved.sort(key=lambda x: x[0], reverse=True)

        for start_idx, end_idx, is_insert, added in resolved:
            if is_insert:
                for i, new_line in enumerate(added):
                    lines.insert(start_idx + 1 + i, new_line)
            else:
                lines[start_idx : end_idx + 1] = added

        return "\n".join(lines)

    def _resolve_tag(
        self, tag: str, hash_map: dict[str, tuple[int, str]]
    ) -> int:
        """Resolve a tag like '2:f1' to a 0-indexed line number."""
        if tag in hash_map:
            return hash_map[tag][0]

        tag_match = re.match(r"^(\d+):", tag)
        if tag_match:
            lineno = int(tag_match.group(1))
            for existing_tag, (idx, _) in hash_map.items():
                if idx == lineno - 1:
                    raise ApplyError(
                        f"Hash mismatch for tag {tag!r}: "
                        f"expected {existing_tag!r}. "
                        f"File may have changed since last read."
                    )

        raise ApplyError(f"Tag not found: {tag!r}")
