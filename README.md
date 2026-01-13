# RLM Extractor

Recursive Language Model (RLM) implementation for schema-based information extraction from arbitrarily long documents.

## Overview

RLM enables LLMs to process documents beyond their native context window limitations. Instead of forcing a massive document into context, RLM treats the document as an external environment that the LLM can programmatically interact with.

### Key Features

| Feature | Description |
|---------|-------------|
| **Unlimited Input Length** | Process documents of any size by chunking |
| **Dual-Model Architecture** | Root LM (orchestrator) + Worker LM (extraction) |
| **Schema-Based Extraction** | Extract structured data using JSON Schema |
| **Text & PDF Support** | Process text documents and extract text from PDFs |
| **Parallel Processing** | Extract from multiple chunks simultaneously |
| **Transparent Failures** | Return all failures with results for manual review |
| **LLM Call Logging** | Automatic logging for debugging and fine-tuning |

## Installation

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or with development dependencies
uv pip install -e ".[dev]"
```

**Quick setup:**
```bash
uv venv && uv pip install -e .
```

## API Key Setup

RLM uses DSPy with [litellm](https://litellm.ai/) for model routing. Set your API keys:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### Supported Providers

| Provider | Environment Variable | Example Models |
|----------|---------------------|----------------|
| **OpenRouter** | `OPENROUTER_API_KEY` | `openrouter/anthropic/claude-sonnet-4` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4`, `anthropic/claude-haiku-4` |

## Quick Start

### Simplest Usage

```python
from rlm_extractor import extract

result = extract(
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"}
        }
    },
    document="John Doe can be reached at john@example.com",
)

print(result.data)  # {'name': 'John Doe', 'email': 'john@example.com'}
```

### With Custom Models

```python
from rlm_extractor import extract, RLMConfig

result = extract(
    schema={
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total": {"type": "number"}
        }
    },
    document=open("invoice.txt").read(),
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    chunk_size=3000,
    max_parallel_workers=10,
)
```

### Full Configuration

```python
from rlm_extractor import RLMExtractor, RLMConfig

config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    chunk_size=2000,
    summary_level="standard",
    max_parallel_workers=5,
    max_turns=20,
)

extractor = RLMExtractor(config)
result = extractor.extract(
    json_schema={"type": "object", "properties": {...}},
    document=open("document.pdf").read(),  # Also supports .txt, .md files
)
```

## How It Works

```
User Input                    Processing                    Output
─────────────────────────────────────────────────────────────────────
JSON Schema  ──►  Convert to YAML  ──►  (internal format)
Document     ──►  Chunk (2K chars)  ──►  INPUT variable

                    Root LM (orchestrator)
                           │
          Parallel First Pass │ Sequential Re-check
                    │        │
         Worker LM (extraction)
         - Full YAML schema
         - Chunk + gist
         - Extract entity contexts
                    │
                    ▼
           Aggregate & Extract Values
           YAML → JSON
           Return + failures
```

**See `logic_loop.md` for the complete orchestration algorithm.**

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_model` | *required* | Model for orchestration (e.g., `openai/gpt-4o`) |
| `worker_text_model` | *required* | Model for text extraction (e.g., `openai/gpt-4o-mini`) |
| `chunk_size` | `2000` | Characters per text chunk |
| `summary_level` | `"standard"` | Chunk gist detail: `minimal`, `standard`, `verbose` |
| `parallel_first_pass` | `True` | Parallel processing for initial extraction |
| `parallel_retry` | `True` | Parallel processing for re-extraction |
| `max_parallel_workers` | `5` | Maximum concurrent API calls |
| `max_turns` | `5` | Maximum orchestration iterations |
| `max_retries` | `3` | Retry attempts per chunk |
| `chunk_timeout` | `300` | Timeout per chunk (seconds) |

## Result Object

```python
@dataclass
class ExtractionResult:
    data: dict                    # Final extracted JSON matching schema
    chunk_gists: list[dict]       # Summary per chunk
    failures: list[dict]          # Failed chunks with errors
    turns: int                    # Number of orchestration iterations
    token_usage: dict             # {root, worker, total}
    log_file_path: str | None     # Path to LLM call log file

    def is_complete(self) -> bool: # True if no failures
    def get_failure_rate(self) -> float:  # 0.0 to 1.0
```

## Examples

```bash
# Set your API key
export OPENROUTER_API_KEY=...

# Run examples
uv run python examples/demo.py
uv run python examples/extract/invoice.py
uv run python examples/extract/contract.py
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_chunker.py

# Run with coverage
uv run pytest --cov=rlm --cov-report=html
```

## Documentation

| Document | Description |
|----------|-------------|
| [`architecture.md`](architecture.md) | System architecture and design principles |
| [`logic_loop.md`](logic_loop.md) | Orchestration algorithm and decision flow |
| [`docs/architecture/architecture.md`](docs/architecture/architecture.md) | Original RLM architecture (code-focused) |
| [`docs/extraction/extraction-design.md`](docs/extraction/extraction-design.md) | Extraction system design |

## Project Status

| Component | Status |
|-----------|--------|
| Core RLM Foundation | ✅ Complete |
| Schema Converter | ✅ Complete |
| Chunker (Text/PDF) | ✅ Complete |
| Worker Extraction | ✅ Complete |
| RLM Extractor | ✅ Complete |
| LLM Call Logging | ✅ Complete |
| Examples | ✅ Complete |
| Tests | ✅ Complete |

## License

MIT

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Original research paper
- [DSPy](https://github.com/stanfordnlp/dspy) - Declarative LM programming framework
