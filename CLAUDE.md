# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLM (Recursive Language Model) is a Python implementation that enables LLMs to process arbitrarily long documents beyond their context window limits. The document is treated as an external environment that the LLM interacts with programmatically via a Python REPL.

**Tech Stack**: Python 3.10+, DSPy (orchestration), litellm (model routing), Pydantic (config)

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
uv run black rlm              # Format (line-length: 100)
uv run ruff check rlm         # Lint
uv run mypy rlm               # Type check
```

## Architecture

### Dual-Model Design

The system uses two distinct LLM roles:

1. **Root LM** (`root_model`) - Orchestrator that plans and coordinates extraction. Always text-based, never sees document content directly.
2. **Worker LM** (`worker_text_model`, `worker_vision_model`) - Performs actual extraction from document chunks. Switches between text and vision models based on input modality.

### Extraction Flow

```
JSON Schema → YAML (internal) → Worker LM
Document → Chunking (2000 chars) → Parallel First Pass
          ↓
Root LM analyzes results → Directs re-extraction of specific chunks
          ↓
Aggregation → YAML → JSON → Return + failures
```

### Key Modules

- `rlm/config.py` - `RLMConfig` with preset functions: `openai_config()`, `anthropic_config()`, `cost_optimized_config()`, `quality_config()`
- `rlm/extract/extractor.py` - `RLMExtractor` main orchestrator
- `rlm/extract/chunker.py` - Strategy pattern for document chunking (text, PDF, images)
- `rlm/repl.py` - Sandboxed Python execution environment for LLM code generation
- `rlm/signatures.py` - DSPy signatures defining LM interfaces

### Configuration Patterns

All models use litellm format with provider prefix: `openai/gpt-4o`, `anthropic/claude-sonnet-4`

DSPy automatically reads API keys from environment variables. Configure via `.env` file:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Design Patterns in Use

- **Strategy Pattern**: Chunking strategies for different document types
- **Template Method**: Extraction workflow with customizable steps
- **Observer Pattern**: Progress tracking through callbacks

## Known Issues

See `issues.md` for details. The main high-severity item is memory risk with large image documents (base64 encoding can cause exhaustion with 100+ 4K images).

## Adding Model Support

When adding support for new models:
1. Add a preset function to `rlm/config.py` following the existing pattern
2. Use litellm model format with provider prefix
3. Ensure vision-capable models are used for `worker_vision_model`
