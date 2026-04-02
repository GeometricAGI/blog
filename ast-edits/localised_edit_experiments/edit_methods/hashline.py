"""Hashline edit method.

Each line is tagged with a short content hash when presented to the LLM.
The LLM references those tags to specify edits, avoiding the need to
reproduce old content verbatim.

Based on the approach described by Can Bölük (oh-my-pi).
"""

from __future__ import annotations

import hashlib
import json
import re

from localised_edit_experiments.data_models import ApplyError, ParseError


def _line_hash(line: str) -> str:
    """Compute a 2-character content hash for a line."""
    return hashlib.md5(line.encode()).hexdigest()[:2]


def tag_lines(code: str) -> str:
    """Tag each line of code with line number and content hash.

    Returns format like:
        1:a3|def hello():
        2:f1|    return "world"
    """
    lines = code.splitlines()
    tagged = []
    for i, line in enumerate(lines, 1):
        h = _line_hash(line)
        tagged.append(f"{i}:{h}|{line}")
    return "\n".join(tagged)


def _build_hash_map(code: str) -> dict[str, tuple[int, str]]:
    """Build mapping from 'lineno:hash' tag to (0-indexed position, content)."""
    lines = code.splitlines()
    result: dict[str, tuple[int, str]] = {}
    for i, line in enumerate(lines):
        h = _line_hash(line)
        tag = f"{i + 1}:{h}"
        result[tag] = (i, line)
    return result


class HashlineJsonOpsMethod:
    """Edit method using content-hashed line references with JSON operations.

    Lines are tagged as LINENO:HASH|content and the LLM references those
    tags to specify replace, insert_after, and delete operations in JSON.
    """

    def system_prompt(self) -> str:
        """Return system prompt for hashline editing."""
        return (
            "You are a code editing assistant. The original code is shown "
            "with each line tagged as `LINENO:HASH|content`, for example:\n\n"
            "```\n"
            "1:a3|def hello():\n"
            '2:f1|    return "world"\n'
            "3:0e|}\n"
            "```\n\n"
            "When asked to edit code, output a JSON array of edit operations "
            "in a ```json fenced block. Each operation references lines by "
            "their `LINENO:HASH` tag.\n\n"
            "Supported operations:\n"
            '- {"op": "replace", "range": ["START_TAG", "END_TAG"], '
            '"content": "new lines"}\n'
            "  Replace lines from START_TAG through END_TAG (inclusive) "
            "with the new content.\n"
            '- {"op": "insert_after", "tag": "TAG", '
            '"content": "new lines to insert"}\n'
            "  Insert new lines after the tagged line.\n"
            '- {"op": "delete", "range": ["START_TAG", "END_TAG"]}\n'
            "  Delete lines from START_TAG through END_TAG (inclusive).\n\n"
            "Rules:\n"
            "- Tags must be copied EXACTLY from the input — both the "
            'line number and the 2-character hash (e.g., "2:f1" not '
            '"2:f2" or just "2"). Do NOT guess or recompute hashes; '
            "copy them verbatim from the tagged code shown above.\n"
            "- Content is the literal new code (with proper indentation)\n"
            "- You may use multiple operations for multiple changes\n"
            "- Make ONLY the change requested\n\n"
            "Example — changing line 2:\n"
            "```json\n"
            '[\n  {"op": "replace", "range": ["2:f1", "2:f1"], '
            '"content": "    return \\"hello\\""}\n]\n'
            "```\n\n"
            "Do not include any explanation outside the JSON block."
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
        """Extract JSON array of hashline operations."""
        pattern = r"```json\s*\n(.*?)```"
        matches = list(re.finditer(pattern, llm_output, re.DOTALL))
        if not matches:
            raise ParseError("No ```json fenced code block found in output")

        # Use the last JSON block — the LLM may self-correct
        match = matches[-1]
        try:
            ops = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e

        if not isinstance(ops, list):
            raise ParseError("Expected a JSON array of operations")

        valid_ops = {"replace", "insert_after", "delete"}
        for op in ops:
            if not isinstance(op, dict) or "op" not in op:
                raise ParseError(f"Invalid operation: {op}")
            if op["op"] not in valid_ops:
                raise ParseError(f"Unknown operation: {op['op']}. Valid: {valid_ops}")
        return ops

    def apply(self, original_code: str, parsed_edit: list[dict]) -> str:
        """Apply hashline operations to original code."""
        lines = original_code.splitlines()
        hash_map = _build_hash_map(original_code)

        # Process operations in reverse order to preserve line positions
        resolved = self._resolve_ops(parsed_edit, hash_map)
        resolved.sort(key=lambda x: x[0], reverse=True)

        for start_idx, end_idx, op in resolved:
            if op["op"] == "replace":
                new_lines = op.get("content", "").splitlines()
                lines[start_idx : end_idx + 1] = new_lines
            elif op["op"] == "delete":
                del lines[start_idx : end_idx + 1]
            elif op["op"] == "insert_after":
                new_lines = op.get("content", "").splitlines()
                for i, new_line in enumerate(new_lines):
                    lines.insert(start_idx + 1 + i, new_line)

        return "\n".join(lines)

    def _resolve_ops(
        self, ops: list[dict], hash_map: dict[str, tuple[int, str]]
    ) -> list[tuple[int, int, dict]]:
        """Resolve tag references to line indices."""
        resolved: list[tuple[int, int, dict]] = []
        for op in ops:
            if op["op"] in ("replace", "delete"):
                tag_range = op.get("range", [])
                if len(tag_range) != 2:
                    raise ApplyError(
                        f"Range must have exactly 2 tags, got: {tag_range}"
                    )
                start_tag, end_tag = tag_range
                start = self._resolve_tag(start_tag, hash_map)
                end = self._resolve_tag(end_tag, hash_map)
                resolved.append((start, end, op))
            elif op["op"] == "insert_after":
                tag = op.get("tag", "")
                idx = self._resolve_tag(tag, hash_map)
                resolved.append((idx, idx, op))
        return resolved

    def _resolve_tag(self, tag: str, hash_map: dict[str, tuple[int, str]]) -> int:
        """Resolve a tag like '2:f1' to a 0-indexed line number."""
        if tag in hash_map:
            return hash_map[tag][0]

        # Fuzzy match: try just the line number if hash doesn't match
        tag_match = re.match(r"^(\d+):", tag)
        if tag_match:
            lineno = int(tag_match.group(1))
            # Check if the line number is valid
            for existing_tag, (idx, _content) in hash_map.items():
                if idx == lineno - 1:
                    raise ApplyError(
                        f"Hash mismatch for tag {tag!r}: "
                        f"expected {existing_tag!r}. "
                        f"File may have changed since last read."
                    )

        raise ApplyError(f"Tag not found: {tag!r}")
