"""Data models for localised code editing experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class EditMethodType(str, Enum):
    """Types of edit methods."""

    WHOLE_FILE = "whole_file"
    UNIFIED_DIFF = "unified_diff"
    SEARCH_REPLACE = "search_replace"
    AST_EDIT = "ast_edit"
    HASHLINE_JSON_OPS = "hashline_json_ops"
    HASHLINE_UNIFIED_DIFF = "hashline_unified_diff"
    HASHLINE_SEARCH_REPLACE = "hashline_search_replace"


class EditDifficulty(str, Enum):
    """Difficulty levels for edit tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EditType(str, Enum):
    """Types of code edits."""

    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    ADD_FEATURE = "add_feature"
    STYLE = "style"


@dataclass
class EditTask:
    """A single code editing benchmark task."""

    task_id: str
    description: str
    original_code: str
    expected_code: str
    difficulty: EditDifficulty
    edit_type: EditType
    test_code: str = ""


@dataclass
class EditAttempt:
    """Result of a single edit attempt."""

    task_id: str
    method: EditMethodType
    model: str
    raw_llm_output: str
    applied_code: str | None
    apply_success: bool
    correct: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    edit_minimality: float
    collateral_damage_score: float = 1.0
    error_message: str | None = None


@dataclass
class ExperimentResults:
    """Collection of all experiment results."""

    attempts: list[EditAttempt] = field(default_factory=list)
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None


class EditError(Exception):
    """Base exception for edit operations."""


class ParseError(EditError):
    """Error parsing LLM output into edit format."""


class ApplyError(EditError):
    """Error applying parsed edit to source code."""
