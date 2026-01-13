# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLM (Recursive Language Model) is a Python implementation that enables LLMs to process arbitrarily long documents beyond their context window limits. The document is treated as an external environment that the LLM interacts with programmatically via a Python REPL.

**Tech Stack**: Python 3.10+, DSPy (orchestration), litellm (model routing), Pydantic (config), pypdf (PDF text extraction)

## Common Commands

**Use `uv` for all package operations:**

```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Testing
uv run pytest                          # All tests
uv run pytest tests/test_chunker.py   # Specific file
uv run pytest --cov=rlm --cov-report=html  # Coverage

# Run examples
uv run python examples/demo.py
uv run python examples/extract/invoice.py
```

**Code quality:**
```bash
uv run black rlm_extractor              # Format (line-length: 100)
uv run ruff check rlm_extractor         # Lint
uv run mypy rlm_extractor               # Type check
```

## Architecture

### Dual-Model Design

The system uses two distinct LLM roles:

1. **Root LM** (`root_model`) - Orchestrator that plans and coordinates extraction. Always text-based, never sees document content directly.
2. **Worker LM** (`worker_text_model`) - Performs actual extraction from text document chunks.

### Extraction Flow

```
JSON Schema → YAML (internal) → Worker LM
Document → Chunking (2000 chars) → Parallel First Pass
          ↓
Workers extract entity contexts (field_name: description)
          ↓
Root LM analyzes contexts → Directs re-extraction of specific chunks
          ↓
Root LM extracts final values from contexts → JSON → Return + failures
```

### Key Modules

- `rlm_extractor/config.py` - `RLMConfig` configuration class
- `rlm_extractor/extract/extractor.py` - `RLMExtractor` main orchestrator
- `rlm_extractor/extract/chunker.py` - Strategy pattern for document chunking (text, PDF)
- `rlm_extractor/extract/processor.py` - Parallel/sequential chunk processing with exponential backoff retry
- `rlm_extractor/extract/schema.py` - JSON ↔ YAML schema conversion
- `rlm_extractor/repl.py` - `REPLState` for persistent state management across RLM turns
- `rlm_extractor/logger.py` - `CallLogger` for LLM call tracking (JSON Lines format)
- `rlm_extractor/signatures.py` - DSPy signatures defining LM interfaces

### Configuration Patterns

All models use litellm format with provider prefix: `openai/gpt-4o`, `anthropic/claude-sonnet-4`, `openrouter/anthropic/claude-sonnet-4`

DSPy automatically reads API keys from environment variables. Configure via `.env` file:
```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Design Patterns in Use

- **Strategy Pattern**: Chunking strategies for different document types
- **Template Method**: Extraction workflow with customizable steps
- **Observer Pattern**: Progress tracking through callbacks

## Known Issues

See `issues.md` for details.

## Adding Model Support

When adding support for new models:
1. Use litellm model format with provider prefix (e.g., `provider/model-name`)
2. Users specify models directly when creating `RLMConfig` or calling `extract()`
