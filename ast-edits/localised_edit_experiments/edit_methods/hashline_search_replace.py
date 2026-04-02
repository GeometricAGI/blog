"""Tag-anchored hashline search-replace method.

Instead of requiring the LLM to reproduce old code verbatim in SEARCH blocks,
the LLM specifies a tag range (e.g., 5:a3..8:0e) to identify lines to replace.
Only the REPLACE content needs to be written out.

Inspired by the approach described in "The Harness Problem" (Can Bölük, 2026).
"""

from __future__ import annotations

import re

from localised_edit_experiments.data_models import ApplyError, ParseError
from localised_edit_experiments.edit_methods.hashline import (
    _build_hash_map,
    tag_lines,
)


class HashlineSearchReplaceMethod:
    """Search-replace using tag ranges instead of verbatim old code."""

    def system_prompt(self) -> str:
        """Return system prompt for tag-anchored search/replace."""
        return (
            "You are a code editing assistant. The original code is shown "
            "with each line tagged as `LINENO:HASH|content`, for example:\n\n"
            "```\n"
            "1:a3|def hello():\n"
            '2:f1|    return "world"\n'
            "3:0e|\n"
            "```\n\n"
            "When asked to edit code, output one or more search-and-replace "
            "blocks using tag ranges to identify the lines to replace:\n\n"
            "<<<SEARCH START_TAG..END_TAG\n"
            ">>>REPLACE\n"
            "replacement text\n"
            "<<<END\n\n"
            "The START_TAG..END_TAG range identifies the lines to replace "
            "(inclusive). Tags must be copied EXACTLY from the input — both "
            "the line number and the 2-character hash.\n\n"
            "To insert new lines AFTER a tagged line (without removing it), "
            "use the same tag for both start and end with the "
            "INSERT_AFTER keyword:\n\n"
            "<<<INSERT_AFTER TAG\n"
            "new lines to insert\n"
            "<<<END\n\n"
            "To delete lines without replacement, use an empty REPLACE:\n\n"
            "<<<SEARCH START_TAG..END_TAG\n"
            ">>>REPLACE\n"
            "<<<END\n\n"
            "Rules:\n"
            "- Tags must be copied EXACTLY from the tagged code (e.g., "
            '"2:f1" not "2:f2" or just "2")\n'
            "- Do NOT reproduce the old code — just specify the tag range\n"
            "- The REPLACE content is the literal new code with proper "
            "indentation\n"
            "- You may use multiple blocks for multiple changes\n"
            "- Do NOT wrap blocks in markdown code fences\n"
            "- Make ONLY the change requested\n\n"
            "Example — replacing line 2:\n\n"
            "<<<SEARCH 2:f1..2:f1\n"
            ">>>REPLACE\n"
            '    return "hello"\n'
            "<<<END\n\n"
            "Example — replacing lines 1 through 2:\n\n"
            "<<<SEARCH 1:a3..2:f1\n"
            ">>>REPLACE\n"
            "def hello(name):\n"
            '    return f"hello {name}"\n'
            "<<<END\n\n"
            "Do not include any explanation outside the blocks."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with hash-tagged code."""
        tagged = tag_lines(original_code)
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```\n{tagged}\n```"
        )

    def parse(self, llm_output: str) -> list[dict]:
        """Extract tag-anchored operations from LLM output."""
        ops: list[dict] = []

        # Match SEARCH range blocks
        search_pattern = (
            r"<<<SEARCH\s+(\d+:[a-f0-9]{2})\.\.(\d+:[a-f0-9]{2})\s*\n"
            r">>>REPLACE\n"
            r"(.*?)"
            r"<<<END"
        )
        for m in re.finditer(search_pattern, llm_output, re.DOTALL):
            content = m.group(3).rstrip("\n")
            ops.append({
                "op": "replace",
                "start_tag": m.group(1),
                "end_tag": m.group(2),
                "content": content,
            })

        # Match INSERT_AFTER blocks
        insert_pattern = (
            r"<<<INSERT_AFTER\s+(\d+:[a-f0-9]{2})\s*\n"
            r"(.*?)"
            r"<<<END"
        )
        for m in re.finditer(insert_pattern, llm_output, re.DOTALL):
            content = m.group(2).rstrip("\n")
            ops.append({
                "op": "insert_after",
                "tag": m.group(1),
                "content": content,
            })

        if not ops:
            raise ParseError(
                "No <<<SEARCH TAG..TAG or <<<INSERT_AFTER TAG blocks found"
            )

        return ops

    def apply(self, original_code: str, parsed_edit: list[dict]) -> str:
        """Apply tag-anchored operations to original code."""
        lines = original_code.splitlines()
        hash_map = _build_hash_map(original_code)

        resolved: list[tuple[int, int, dict]] = []
        for op in parsed_edit:
            if op["op"] == "replace":
                start = self._resolve_tag(op["start_tag"], hash_map)
                end = self._resolve_tag(op["end_tag"], hash_map)
                resolved.append((start, end, op))
            elif op["op"] == "insert_after":
                idx = self._resolve_tag(op["tag"], hash_map)
                resolved.append((idx, idx, op))

        # Process in reverse order to preserve positions
        resolved.sort(key=lambda x: x[0], reverse=True)

        for start_idx, end_idx, op in resolved:
            if op["op"] == "replace":
                content = op.get("content", "")
                new_lines = content.splitlines() if content else []
                lines[start_idx : end_idx + 1] = new_lines
            elif op["op"] == "insert_after":
                new_lines = op.get("content", "").splitlines()
                for i, new_line in enumerate(new_lines):
                    lines.insert(start_idx + 1 + i, new_line)

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
