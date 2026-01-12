# RLM Extractor

Recursive Language Model (RLM) implementation with schema-based information extraction.

## Overview

RLM enables LLMs to process **arbitrarily long documents** beyond their native context window limitations. Instead of forcing a massive document into context, RLM treats the document as an external environment that the LLM can programmatically interact with.

### Key Features

- **Unlimited Input Length** - Process documents of any size by chunking
- **Dual-Model Architecture** - Root LM (orchestrator) + Worker LM (extraction)
- **Schema-Based Extraction** - Extract structured data using JSON Schema
- **Text & PDF Support** - Process text documents and extract text from PDFs
- **Parallel Processing** - Extract from multiple chunks simultaneously
- **Transparent Failures** - Return all failures with results for manual review
- **LLM Call Logging** - Automatic logging of all Root/Worker LM calls for fine-tuning

## PDF Support

RLM uses [pypdf](https://pypdf.readthedocs.io/) for text extraction from PDFs. pypdf is a pure-Python library that works on all platforms including Windows, macOS, and Linux. No external dependencies required.

Simply pass a PDF file path and RLM will automatically extract text and process it.

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

RLM uses DSPy which internally uses [litellm](https://litellm.ai/) to automatically handle API keys from environment variables. Set your API keys before running:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys, then source it
export $(cat .env | xargs)

# Or export directly (not recommended for production)
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| **OpenRouter (Recommended)** | `OPENROUTER_API_KEY` | Various (see [openrouter.ai](https://openrouter.ai/models)) |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4`, `anthropic/claude-haiku-4` |

> **Note:** DSPy/litellm will automatically resolve the correct API key based on the model prefix. You only need to set environment variables for the providers you plan to use.

## Simplest Usage

The simplest way to use rlm_extractor is with the `extract()` function:

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

**Default models used by `extract()`:**
- Root LM: `openrouter/minimax/minimax-m2.1`
- Worker LM: `openrouter/google/gemini-2.5-flash-lite`

You can override these defaults by passing your own model parameters:

```python
result = extract(
    schema={...},
    document=document,
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
)
```

### Customizing with Config Parameters

You can override any config parameter:

```python
from rlm_extractor import extract

result = extract(
    schema={...},
    document=large_document,
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    chunk_size=3000,  # Larger chunks
    max_parallel_workers=10,  # More parallelism
    max_turns=30  # More extraction rounds
)
```

### Customizing with Config Parameters

You can override any config parameter:

```python
from rlm_extractor import extract

result = extract(
    schema={...},
    document=large_document,
    chunk_size=3000,  # Larger chunks
    max_parallel_workers=10,  # More parallelism
    max_turns=30  # More extraction rounds
)
```

## Quick Start

```python
from rlm_extractor import RLMExtractor, RLMConfig

# Configure with your preferred models
config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
)

# Initialize extractor
extractor = RLMExtractor(config)

# Define extraction schema
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total": {"type": "number"}
    }
}

# Extract from document
result = extractor.extract(
    json_schema=schema,
    document=open("invoice.txt").read()
)

# Access results
print(result.data)  # Extracted JSON
print(result.failures)  # Any failures
print(result.log_file_path)  # Path to LLM call log file (JSON Lines format)
```

## LLM Call Logging

RLM automatically logs all LLM requests and responses to a JSON Lines (`.jsonl`) file. This is useful for:
- Debugging extraction issues
- Building training datasets for fine-tuning
- Analyzing model behavior
- Tracking token usage and performance

### Log File Location

Log files are created in the current working directory with the format:
```
rlm_extraction_YYYYMMDD_HHMMSS.jsonl
```

The log file path is available in `result.log_file_path` after extraction completes.

### Log Format

Each line is a complete JSON object representing one LLM interaction:

**Request entry (Root LM):**
```json
{
  "timestamp": "2024-01-11T12:34:56.789Z",
  "call_id": "uuid-here",
  "lm_type": "root",
  "model": "openrouter/anthropic/claude-sonnet-4",
  "signature": "RootExtractionSignature",
  "call_type": "root_decision",
  "stage": "request",
  "request": {
    "task": "Extract invoice fields",
    "field_completion": "invoice_number: found, date: missing"
  },
  "metadata": {
    "turn": 1
  }
}
```

**Request entry (Worker LM):**
```json
{
  "timestamp": "2024-01-11T12:35:12.456Z",
  "call_id": "uuid-here",
  "lm_type": "worker",
  "model": "openrouter/anthropic/claude-haiku-4",
  "signature": "WorkerExtractionSignature",
  "call_type": "worker_extraction",
  "stage": "request",
  "request": {
    "yaml_schema": "invoice_number: string\\ndate: string",
    "chunk_idx": 0,
    "chunk_content": "This is chunk 0 content...",
    "condensed_guidance": "Extract all fields",
    "targeted_prompt": ""
  },
  "metadata": {
    "chunk_idx": 0,
    "attempt": 1,
    "max_attempts": 2
  }
}
```

**Response entry:**
```json
{
  "timestamp": "2024-01-11T12:35:15.789Z",
  "call_id": "uuid-here",
  "stage": "response",
  "response": {
    "thought": "Date field is missing, need to re-extract",
    "action": "re_extract",
    "target_chunk": "2"
  },
  "metadata": {}
}
```

**Error entry:**
```json
{
  "timestamp": "2024-01-11T12:35:20.123Z",
  "call_id": "uuid-here",
  "stage": "error",
  "error": "Timeout after 2 attempts",
  "metadata": {
    "chunk_idx": 0,
    "attempt": 2,
    "error_type": "timeout"
  }
}
```

### Log Entry Fields

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp when entry was created |
| `call_id` | UUID to pair request/response entries |
| `lm_type` | `"root"` or `"worker"` - indicates which LM made the call |
| `model` | Full model identifier (e.g., `openrouter/anthropic/claude-sonnet-4`) |
| `signature` | DSPy signature class name |
| `call_type` | Specific operation type (root_decision, worker_extraction, context_condensation) |
| `stage` | `"request"`, `"response"`, or `"error"` |
| `request` | Full request payload (for `stage: "request"`) |
| `response` | Full response payload (for `stage: "response"`) |
| `error` | Error message (for `stage: "error"`) |
| `metadata` | Context information (chunk_idx, turn, attempt, error_type, etc.) |

### Using Logs for Fine-Tuning

The log files are in JSON Lines format, making them easy to parse and convert to training datasets:

```python
import json

# Read log file
with open(result.log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)

        # Filter for worker LM extractions only
        if (
            entry["stage"] == "request"
            and entry["lm_type"] == "worker"
            and entry["call_type"] == "worker_extraction"
        ):
            call_id = entry["call_id"]
            request = entry["request"]

            # Find corresponding response
            # (you'd need to read all lines into memory or stream twice)
            # ...

# Example: Extract all successful worker extractions
successful_extractions = []
lines = []

with open(result.log_file_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# Pair requests and responses
for entry in lines:
    if entry["stage"] == "request" and entry["lm_type"] == "worker":
        call_id = entry["call_id"]
        response = next(
            (
                e
                for e in lines
                if e["call_id"] == call_id and e["stage"] == "response"
            ),
            None,
        )
        if response:
            successful_extractions.append(
                {
                    "request": entry["request"],
                    "response": response["response"],
                    "model": entry["model"],
                }
            )

print(f"Found {len(successful_extractions)} successful extractions")
```
```

## Examples

See `examples/` for complete examples:

- `examples/demo.py` - Simple demo
- `examples/extract/invoice.py` - Invoice extraction
- `examples/extract/contract.py` - Contract extraction

```bash
# Run examples with uv
export OPENROUTER_API_KEY=...  # or OPENAI_API_KEY, ANTHROPIC_API_KEY
uv run python examples/demo.py
uv run python examples/extract/invoice.py
```

## Configuration

### Sample Configurations

```python
from rlm_extractor import RLMConfig

# OpenRouter models (recommended)
config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
)

# OpenAI models
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
)

# Anthropic models
config = RLMConfig(
    root_model="anthropic/claude-sonnet-4",
    worker_text_model="anthropic/claude-haiku-4",
)
```

### Custom Config Parameters

```python
config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    chunk_size=2000,
    summary_level="standard",  # minimal/standard/verbose
    max_parallel_workers=5,
)
```

## How It Works

1. **Schema Conversion** - JSON Schema is converted to YAML (internal)
2. **Document Chunking** - Split into fixed-size chunks at nearest space
3. **Parallel First Pass** - Workers extract from chunks in parallel
4. **Root Orchestration** - Root LM analyzes results and directs re-extraction
5. **Aggregation** - Results are merged and converted back to JSON

```
User Input                    Internal Processing              Output
─────────────────────────────────────────────────────────────────────
JSON Schema  ──►  Auto-convert to YAML  ──►  (internal)
Document     ──►  Chunk (2K chars)        ──►  INPUT variable

                    Root LM (orchestrator)
                           │
          Parallel First Pass │ Sequential Re-check
                    │        │
         Worker LM (text)
         - Full YAML schema
         - Chunk + gist
         - Extract fields
                    │
                    ▼
           Aggregate Results
           YAML → JSON
           Return + failures
```

## Testing

```bash
# Run all tests (with uv)
uv run pytest

# Run specific test file
uv run pytest tests/test_chunker.py

# Run with coverage
uv run pytest --cov=rlm --cov-report=html
```

## Documentation

- `docs/README.md` - Documentation index
- `docs/research/` - Original papers and study
- `docs/architecture/` - System architecture
- `docs/plans/` - Implementation plans
- `docs/extraction/` - Extraction system design

## Project Status

| Component | Status |
|-----------|--------|
| Core RLM Foundation | ✅ Complete |
| Schema Converter | ✅ Complete |
| Chunker | ✅ Complete |
| Worker Extraction | ✅ Complete |
| RLM Extractor | ✅ Complete |
| LLM Call Logging | ✅ Complete |
| Examples | ✅ Complete |
| Tests | ✅ Complete |

## License

MIT

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Original research paper
