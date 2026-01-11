# RLM Extractor

Recursive Language Model (RLM) implementation with schema-based information extraction.

## Overview

RLM enables LLMs to process **arbitrarily long documents** beyond their native context window limitations. Instead of forcing a massive document into context, RLM treats the document as an external environment that the LLM can programmatically interact with.

### Key Features

- **Unlimited Input Length** - Process documents of any size by chunking
- **Dual-Model Architecture** - Root LM (orchestrator) + Worker LM (extraction)
- **Schema-Based Extraction** - Extract structured data using JSON Schema
- **Text & Image Support** - Process text documents or document images
- **Parallel Processing** - Extract from multiple chunks simultaneously
- **Transparent Failures** - Return all failures with results for manual review

# PDF Support

RLM uses [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing. PyMuPDF is a high-performance Python library that works on all platforms including Windows, macOS, and Linux. No external dependencies required.

## PDF Processing Mode

The `pdf_mode` parameter in `extract()` and `RLMExtractor` controls how PDFs are processed:

- `'text'` (recommended for text-based PDFs): Extracts text directly from PDF and processes as text chunks
- `'image'` (recommended for scanned PDFs): Renders PDF pages as images for vision models
- `'auto'` (default): Chooses 'text' for text/markdown files, 'image' for PDFs

**Default behavior**:
- Text/markdown files → text mode
- PDF files → image mode
- Image files → vision mode
- Image paths → vision mode
- Lists → vision mode

**User override**:
Pass `pdf_mode="text"` to force text extraction from PDFs.

Example:
```python
from rlm_extractor import RLMExtractor, RLMConfig

config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
)
extractor = RLMExtractor(config)
result = extractor.extract(
    json_schema=...,
    document="invoice.pdf",  # PDF file
    pdf_mode="image"  # Force image mode instead
)
```

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

### PDF Support

RLM uses [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing. PyMuPDF is a high-performance Python library that works on all platforms including Windows, macOS, and Linux. No external dependencies required.

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
- Worker text/vision LM: `openrouter/google/gemini-2.5-flash-lite`

You can override these defaults by passing your own model parameters:

```python
result = extract(
    schema={...},
    document=document,
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
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
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
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
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
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
```

## Examples

See `examples/` for complete examples:

- `examples/demo.py` - Simple demo
- `examples/extract/invoice.py` - Invoice extraction
- `examples/extract/contract.py` - Contract extraction
- `examples/extract/images.py` - Image document extraction

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
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
)

# OpenAI models
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
    worker_vision_model="openai/gpt-4o",
)

# Anthropic models
config = RLMConfig(
    root_model="anthropic/claude-sonnet-4",
    worker_text_model="anthropic/claude-haiku-4",
    worker_vision_model="anthropic/claude-sonnet-4",
)
```

### Custom Config Parameters

```python
config = RLMConfig(
    root_model="openrouter/anthropic/claude-sonnet-4",
    worker_text_model="openrouter/anthropic/claude-haiku-4",
    worker_vision_model="openrouter/anthropic/claude-sonnet-4",
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
         Worker LM (text/vision)
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
| Examples | ✅ Complete |
| Tests | ✅ Complete |

## License

MIT

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Original research paper
