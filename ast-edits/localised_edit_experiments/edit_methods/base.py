"""Base protocol for edit methods."""

from __future__ import annotations

from typing import Protocol


class EditMethod(Protocol):
    """Protocol for code edit methods."""

    def system_prompt(self) -> str:
        """Return the system prompt instructing the LLM on edit format."""
        ...

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build the user message with original code and edit instruction."""
        ...

    def parse(self, llm_output: str) -> object:
        """Extract structured edit from LLM response."""
        ...

    def apply(self, original_code: str, parsed_edit: object) -> str:
        """Apply parsed edit to original code. Raises on failure."""
        ...
