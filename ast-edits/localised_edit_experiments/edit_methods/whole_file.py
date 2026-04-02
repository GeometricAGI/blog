"""Whole-file rewrite edit method."""

from __future__ import annotations

import re

from localised_edit_experiments.data_models import ApplyError, ParseError


class WholeFileMethod:
    """Edit method that returns the entire updated file."""

    def system_prompt(self) -> str:
        """Return system prompt for whole-file rewrite."""
        return (
            "You are a code editing assistant. When asked to edit code, "
            "return the ENTIRE updated file in a single ```python fenced "
            "code block. Do not include any explanation outside the code block. "
            "Do not omit any part of the file - return the complete file.\n\n"
            "CRITICAL: Make ONLY the change requested. Every line of code "
            "that does not need to change must be reproduced EXACTLY as it "
            "appears in the original — same whitespace, same quotes, same "
            "formatting, same variable names. Do not reformat, restyle, "
            "rename variables, reorder imports, collapse multi-line "
            "expressions onto one line, or make ANY modifications beyond "
            "what the instruction asks for."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with original code and instruction."""
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```python\n{original_code}\n```"
        )

    def parse(self, llm_output: str) -> str:
        """Extract code from ```python fenced block."""
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if not match:
            raise ParseError("No ```python fenced code block found in output")
        return match.group(1).rstrip("\n")

    def apply(self, original_code: str, parsed_edit: str) -> str:
        """Apply whole-file edit (trivial replacement)."""
        if not parsed_edit.strip():
            raise ApplyError("Parsed edit is empty")
        return parsed_edit
