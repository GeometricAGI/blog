# AST Edits: The Code Editing Format Nobody Uses

Code to reproduce the results from [AST Edits: The Code Editing Format Nobody Uses](https://www.geometric.dev/blog/ast-edits).

## Overview

This benchmark compares 7 code editing formats used by AI coding tools:

| Method | Description |
|--------|-------------|
| **Whole file** | Model regenerates the entire file |
| **Search/replace** | Exact-match find/replace blocks |
| **Unified diff** | Standard `@@` hunk-based diffs |
| **AST edit** | AST-targeted operations on functions/classes by name |
| **Hashline JSON ops** | JSON operations referencing content-hashed line tags |
| **Hashline search/replace** | Tag-anchored search/replace (no verbatim old code) |
| **Hashline unified diff** | Tag-anchored diff (no context line reproduction) |

29 editing tasks range from 100-line to 4,200-line Python files, covering bug fixes, feature additions, and multi-site refactors.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set API keys for the models you want to test in a `.env` file:

```
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

## Usage

Run all 7 methods with a single model:

```bash
uv run run-edit-experiment --model claude-haiku-4-5-20251001
```

Run specific methods:

```bash
uv run run-edit-experiment \
  --model claude-haiku-4-5-20251001 \
  --method ast_edit \
  --method whole_file \
  --method search_replace
```

Test multiple models:

```bash
uv run run-edit-experiment \
  --model claude-haiku-4-5-20251001 \
  --model openai/o4-mini \
  --model openai/gpt-5.4 \
  --model claude-opus-4-6 \
  --output-dir results
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `claude-haiku-4-5-20251001` | LLM to test (repeatable) |
| `--method` | all 7 | Edit method to test (repeatable) |
| `--output-dir` | `results` | Where to write results |
| `--sleep` | `1.0` | Seconds between API calls |

Models are routed through [litellm](https://docs.litellm.ai/), so any supported provider works.

## Output

Results are written to the output directory:

- `results.json` — raw data for every attempt
- `results.md` — summary tables (correctness, token usage, minimality, collateral damage)
- `success_rates.png`, `token_usage.png`, `latency.png`, `edit_minimality.png`, `difficulty_by_model.png` — plots
