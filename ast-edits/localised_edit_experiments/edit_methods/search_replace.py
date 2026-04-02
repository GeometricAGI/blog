"""Search-replace edit method."""

from __future__ import annotations

import re

from localised_edit_experiments.data_models import ApplyError, ParseError


class SearchReplaceMethod:
    """Edit method using exact-match search/replace blocks."""

    def system_prompt(self) -> str:
        """Return system prompt for search/replace."""
        return (
            "You are a code editing assistant. When asked to edit code, "
            "output one or more search-and-replace blocks in this exact format:\n\n"
            "<<<SEARCH\n"
            "exact text to find\n"
            ">>>REPLACE\n"
            "replacement text\n"
            "<<<END\n\n"
            "Rules:\n"
            "- The SEARCH text must match the original file exactly, "
            "including whitespace, indentation, and newlines\n"
            "- The SEARCH text must appear exactly once in the file\n"
            "- Include enough context lines in SEARCH to be unique\n"
            "- You may use multiple blocks for multiple changes\n"
            "- Do NOT wrap blocks in markdown code fences\n"
            "- Do not include any explanation outside the blocks\n\n"
            "Example with multi-line match:\n\n"
            "<<<SEARCH\n"
            "def foo(x):\n"
            "    return x + 1\n"
            ">>>REPLACE\n"
            "def foo(x):\n"
            "    return x + 2\n"
            "<<<END\n\n"
            "Make ONLY the change requested."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with original code and instruction."""
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```python\n{original_code}\n```"
        )

    def parse(self, llm_output: str) -> list[tuple[str, str]]:
        """Extract (old_str, new_str) tuples from search/replace blocks."""
        pattern = r"<<<SEARCH\n(.*?)>>>REPLACE\n(.*?)<<<END"
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if not matches:
            raise ParseError("No <<<SEARCH...>>>REPLACE...<<<END blocks found")

        pairs: list[tuple[str, str]] = []
        for match_old, match_new in matches:
            pairs.append((match_old.rstrip("\n"), match_new.rstrip("\n")))
        return pairs

    def apply(self, original_code: str, parsed_edit: list[tuple[str, str]]) -> str:
        """Apply search/replace pairs sequentially."""
        result = original_code
        for old_str, new_str in parsed_edit:
            count = result.count(old_str)
            if count == 0:
                raise ApplyError(
                    f"Search string not found in code: {old_str[:80]!r}..."
                )
            if count > 1:
                raise ApplyError(
                    f"Search string found {count} times (must be unique): "
                    f"{old_str[:80]!r}..."
                )
            result = result.replace(old_str, new_str, 1)
        return result
