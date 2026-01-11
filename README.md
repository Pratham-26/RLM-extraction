# RLM Implementation

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
from rlm import RLMExtractor, openai_config

extractor = RLMExtractor(openai_config())
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
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4`, `anthropic/claude-haiku-4` |
| OpenRouter | `OPENROUTER_API_KEY` | Various (see [openrouter.ai](https://openrouter.ai/models)) |

> **Note:** DSPy/litellm will automatically resolve the correct API key based on the model prefix. You only need to set environment variables for the providers you plan to use.

## Quick Start

```python
from rlm import RLMExtractor, RLMConfig

# Configure (using preset)
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
    worker_vision_model="openai/gpt-4o",
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
export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY
uv run python examples/demo.py
uv run python examples/extract/invoice.py
```

## Configuration

### Preset Configs

```python
from rlm.config import openai_config, anthropic_config, cost_optimized_config, quality_config

# OpenAI models
config = openai_config()

# Anthropic models
config = anthropic_config()

# Cost-optimized
config = cost_optimized_config()

# Maximum quality
config = quality_config()
```

### Custom Config

```python
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
    worker_vision_model="openai/gpt-4o",
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
