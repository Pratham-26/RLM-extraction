# RLM Schema Extraction System - Design Document

**Status**: Design Complete | Implementation Pending
**Created**: 2025-01-10
**Authors**: RLM Implementation Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [REPL State Management](#repl-state-management)
6. [Error Handling](#error-handling)
7. [Project Structure](#project-structure)
8. [API Reference](#api-reference)
9. [Usage Examples](#usage-examples)
10. [Implementation Phases](#implementation-phases)
11. [Configuration Reference](#configuration-reference)

---

## Overview

### Purpose

Build a schema-based information extraction system using Recursive Language Model (RLM) concepts. The system extracts structured data from arbitrarily large documents (text or images) according to a user-provided JSON Schema.

### Key Innovation

Traditional approaches must fit the entire document into the LLM context window. RLM Extraction treats the document as an external environment (`INPUT` variable) that the LLM can programmatically interact with via Python code execution and parallel worker calls.

### Design Principles

1. **Unlimited Input Length** - Process documents of any size by chunking
2. **Dual-Model Architecture** - Root LM (orchestrator) + Worker LM (extraction)
3. **Parallel First Pass** - Extract from multiple chunks simultaneously
4. **Sequential Re-examination** - Targeted follow-up extraction when needed
5. **YAML as Implementation Detail** - Convert JSON Schema to YAML for LM comprehension
6. **Configurable Parallelism** - Control concurrent API calls
7. **Transparent Failures** - Return all failures with results for manual review

---

## Architecture

### System Diagram

```
User Input                    Internal Processing              Output
─────────────────────────────────────────────────────────────────────
JSON Schema  ──►  Auto-convert to YAML  ──►  (workers never see)
                 Split into YAML chunks

Document (text)  ──►  INPUT = str (2K char chunks)
Document (images) ──► INPUT = List[base64]

                    ┌─────────────────────────────────┐
                    │   Root LM (GPT-4o / Claude)     │
                    │   - Orchestrates extraction     │
                    │   - Sees: gists, extracted      │
                    │   - Directs re-examination      │
                    └─────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          Parallel First Pass                    Sequential Re-check
                    │                                   │
          Worker LM (text)                 Worker LM (text/vision)
          - Full YAML schema               - Targeted prompt
          - Chunk + gist                   - Specific chunk
          - Extract fields                 - Extract specific field
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                           Aggregate Results
                           YAML → JSON
                           Return + failures
```

### Dual-Model Routing

| Input Modality | Root LM | Worker LM |
|----------------|---------|-----------|
| Text documents | Text model (GPT-4o, Claude Sonnet) | Text model (GPT-4o-mini, Claude Haiku) |
| Images | Text model | Vision model (GPT-4o, Claude Sonnet) |

The Root LM is always a text model (orchestrates via reasoning). Worker LM switches based on input type.

---

## Component Design

### RLMConfig

Configuration for dual LM setup with modality routing:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RLMConfig:
    """Configuration for RLM Schema Extraction."""

    # Model configuration
    root_model: str              # Orchestrator (always text-based)
    worker_text_model: str       # For text documents

    # Chunking
    chunk_size: int = 2000       # Configurable chunk size (chars)

    # Worker behavior
    summary_level: Literal["minimal", "standard", "verbose"] = "standard"

    # Parallelism control
    parallel_first_pass: bool = True
    parallel_retry: bool = False
    max_parallel_workers: int = 5  # Concurrent requests at once

    # Execution limits
    max_turns: int = 20
    code_execution_timeout: int = 30

    # API configuration
    api_key: str | None = None
```

### SchemaConverter

Internal JSON ↔ YAML transformation (users never see YAML):

```python
class SchemaConverter:
    """Convert between JSON Schema and YAML for LM processing."""

    def json_to_yaml_chunks(self, json_schema: dict) -> str:
        """
        Convert JSON Schema to YAML format optimized for LM comprehension.

        - Flatten nested structures with clear comments
        - Add type hints and descriptions
        - Keep as single YAML string (workers can handle full schema)
        """

    def yaml_to_json(self, extracted_yaml: str, original_schema: dict) -> dict:
        """
        Convert extracted YAML back to matching JSON structure.

        - Validates against original schema
        - Handles missing optional fields
        - Type conversion (strings to numbers/booleans as needed)
        """
```

### Chunker

Document slicing for text and images:

```python
from typing import Union
from PIL import Image
import base64

class Chunker:
    """Split documents into processable chunks."""

    def chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Fixed-size slices at nearest space.

        - Splits at chunk_size characters
        - Adjusts to nearest space (doesn't cut words)
        - Returns list of text chunks
        """

    def encode_images(self, images: list[Image.Image]) -> list[str]:
        """
        Convert PIL Images to base64 strings for INPUT.

        - Returns list of base64-encoded images
        - Each image becomes one "chunk" for iteration
        """

    def chunk_text_document(self, text: str, chunk_size: int) -> list[dict]:
        """
        Return chunk metadata with content.

        Returns:
            [{idx: 0, content: "...", start: 0, end: 2000}, ...]
        """
```

### DSPy Signatures

Root LM signature for orchestration:

```python
import dspy

class RootExtractionSignature(dspy.Signature):
    """Root LM reasoning for schema extraction orchestration."""

    task = dspy.InputField(desc="Extract according to schema")
    trajectory = dspy.InputField(desc="History: gists, extracted, failures")
    yaml_schema = dspy.InputField(desc="Full YAML schema")
    state_summary = dspy.InputField(desc="Completed chunks, results so far")

    thought = dspy.OutputField(desc="Reasoning about next action")
    action = dspy.OutputField(desc="extract_chunk, re_extract, or finalize")
    target_chunk = dspy.OutputField(desc="Which chunk to process (int or None)")
    targeted_prompt = dspy.OutputField(desc="Specific prompt for re-extraction or None")
```

Worker LM signature for extraction:

```python
class WorkerExtractionSignature(dspy.Signature):
    """Worker LM extraction from a single chunk."""

    yaml_schema = dspy.InputField(desc="Full YAML schema")
    chunk_content = dspy.InputField(desc="Document chunk text or base64 image")
    chunk_idx = dspy.InputField(desc="Chunk index for reference")
    targeted_prompt = dspy.InputField(desc="Optional specific instruction", default="")

    gist = dspy.OutputField(desc="Page/chunk summary for Root LM (2-3 sentences)")
    extracted = dspy.OutputField(desc="Extracted YAML fields (partial schema)")
    confidence = dspy.OutputField(desc="high/medium/low - extraction confidence")
    missing_fields = dspy.OutputField(desc="List of schema fields not found in this chunk")
```

### RLMExtractor

Main orchestrator module:

```python
class RLMExtractor(dspy.Module):
    """Schema-based extraction using RLM paradigm."""

    def __init__(self, config: RLMConfig):
        """
        Initialize with dual LM configuration.

        - Sets up Root LM (always text model)
        - Sets up Worker LM (text or vision based on input)
        - Initialize DSPy signatures
        - Create worker pool
        """

    def extract(
        self,
        json_schema: dict,
        document: Union[str, list[Image.Image]],
        task: str | None = None
    ) -> ExtractionResult:
        """
        Main extraction entry point.

        Args:
            json_schema: JSON Schema for extraction
            document: Text string or list of PIL Images
            task: Optional custom task description

        Returns:
            ExtractionResult with data, gists, failures, usage
        """

    def _detect_modality(self, document) -> Literal["text", "vision"]:
        """Determine if we need vision workers."""

    def _route_worker_lm(self, modality: str) -> dspy.LM:
        """Return appropriate worker LM for input type."""
```

---

## Data Flow

### Text Mode Flow

```
1. User calls:
   extractor = RLMExtractor(config)
   result = extractor.extract(
       json_schema={
           "type": "object",
           "properties": {
               "invoice_number": {"type": "string"},
               "date": {"type": "string"},
               "line_items": {
                   "type": "array",
                   "items": {
                       "description": {"type": "string"},
                       "quantity": {"type": "integer"},
                       "price": {"type": "number"}
                   }
               }
           }
       },
       document=open("huge_invoice.txt").read()  # millions of chars
   )

2. Internal initialization:
   INPUT = document  # Store in REPL
   yaml_schema = schema_converter.json_to_yaml_chunks(json_schema)
   chunks = chunker.chunk_text(document, config.chunk_size)
   # [{"idx": 0, "content": "...", "start": 0, "end": 2000}, ...]

3. Parallel First Pass:
   executor = ThreadPoolExecutor(max_workers=config.max_parallel_workers)
   futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

   Each chunk processor:
       worker = WorkerExtractionSignature
       result = worker(
           yaml_schema=yaml_schema,
           chunk_content=chunk["content"],
           chunk_idx=chunk["idx"],
           targeted_prompt=""
       )
       → Returns: {gist, extracted, confidence, missing_fields}

   Store in REPLState:
       chunk_summaries = [
           {idx: 0, gist: "Page contains header and line items 1-5", confidence: "high"},
           {idx: 1, gist: "Page continues with line items 6-12", confidence: "high"},
           ...
       ]
       results_so_far = merge_all(extracted)
       completed_chunks = [0, 1, 2, ...]

4. Root LM Analysis (orchestration loop):
   for turn in range(config.max_turns):
       root = RootExtractionSignature
       root_result = root(
           task=task or "Extract according to schema",
           trajectory=format_trajectory(),
           yaml_schema=yaml_schema,
           state_summary=f"{len(completed_chunks)}/{total_chunks} complete"
       )

       match root_result.action:
           case "finalize":
               break
           case "re_extract":
               # Sequential targeted extraction
               chunk = chunks[root_result.target_chunk]
               result = process_chunk_sequential(
                   chunk,
                   targeted_prompt=root_result.targeted_prompt
               )
               update_state(result)

5. Finalize:
   final_json = schema_converter.yaml_to_json(results_so_far, json_schema)
   return ExtractionResult(
       data=final_json,
       chunk_gists=repl.chunk_summaries,
       failures=list(repl.failed_chunks.values()),
       turns=turn,
       token_usage=collect_usage()
   )
```

### Image Mode Flow

Same structure, with differences:

- `INPUT = [base64_img_0, base64_img_1, ...]` (not a string)
- `chunker.encode_images(images)` creates base64 list
- `chunk_content` is base64 string, not text
- Vision workers can directly "see" images without OCR

---

## REPL State Management

### REPLState Class

```python
class REPLState:
    """Persistent state across RLM extraction turns."""

    # The document (never seen in full by Root LM)
    INPUT: Union[str, list[str]]  # text OR base64 images

    # Chunk tracking
    total_chunks: int
    completed_chunks: set[int]
    failed_chunks: dict[int, str]  # {chunk_idx: error_message}

    # What Root LM sees (not raw INPUT)
    chunk_summaries: list[dict]  # [{idx, gist, confidence, fields_found}]
    results_so_far: dict  # Accumulated extracted data (YAML format)

    # Configuration
    yaml_schema: str  # Full YAML schema for workers
    summary_level: str  # minimal/standard/verbose

    def get_pending_chunks(self) -> list[int]:
        """Return chunks not yet processed."""

    def update_chunk_result(
        self,
        idx: int,
        gist: str,
        extracted: dict,
        confidence: str
    ):
        """Called after worker completes successfully."""

    def merge_extracted(self, extracted: dict):
        """Merge new extraction into results_so_far."""

    def mark_failed(self, idx: int, error: str):
        """Record a chunk failure."""
```

### What Root LM Sees Each Turn

```
Task: "Extract according to the following YAML schema:
invoice_details:
  invoice_number: string pattern "INV-####"
  date: string (ISO format)
  vendor: object
    name: string
    address: string
  line_items: array
    - description: string
      quantity: integer
      price: number

Trajectory:
Turn 1: Launched parallel extraction on 50 chunks
Turn 2: Received 48 successful results, 2 failures (retried)
Turn 3: All chunks processed. Summary below.
Turn 4: Re-examining chunk 17 - need invoice number specifically

Current State:
- Completed: 50/50 chunks
- Results so far:
  invoice_details:
    vendor:
      name: "Acme Corp"
    line_items: [...] (15 items extracted)
- Missing fields: invoice_number, date
- Chunk summaries (high confidence for schema fields):
  Chunk 0: "Header page with vendor info"
  Chunk 17: "Contains invoice number and date in header"
  Chunk 23: "FAILED - timeout"
  Chunk 45: "FAILED - malformed output"
```

### What Workers Receive

```
First pass (parallel):
  yaml_schema: "<full YAML schema>"
  chunk_content: "<2000 chars of document>" OR "<base64 image>"
  chunk_idx: 17
  targeted_prompt: None

Re-extraction (sequential):
  yaml_schema: "<full YAML schema>"
  chunk_content: "<same chunk content>"
  chunk_idx: 17
  targeted_prompt: "Extract the invoice_number field specifically.
                    Look for patterns like INV-#### or Invoice: #####"
```

---

## Error Handling

### Retry Strategy

One automatic retry for consecutive failures:

```python
class ChunkProcessor:
    """Process individual chunks with retry logic."""

    def process_chunk(
        self,
        idx: int,
        content: str,
        yaml_schema: str,
        targeted_prompt: str = ""
    ) -> dict:
        """
        Process a chunk with automatic retry.

        Returns:
            {success: bool, data: dict | error: str}
        """
        attempts = 0
        max_attempts = 2  # initial + 1 retry

        while attempts < max_attempts:
            try:
                result = self.worker_lm(
                    yaml_schema=yaml_schema,
                    chunk_content=content,
                    targeted_prompt=targeted_prompt
                )
                return {"success": True, "data": result}

            except TimeoutError:
                attempts += 1
                if attempts >= max_attempts:
                    raise
                # Retry without modifying prompt for timeout

            except YAMLParseError as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise
                # Retry with instruction to fix YAML
                targeted_prompt = (
                    f"Please fix your YAML output. Previous attempt was invalid: {str(e)[:100]}"
                )

            except APIError as e:
                if e.retryable:
                    time.sleep(2 ** attempts)  # exponential backoff
                    continue
                raise  # non-retryable API errors
```

### Failure Tracking

```python
@dataclass
class ExtractionResult:
    """Result of schema extraction."""

    data: dict  # Final JSON result matching schema
    chunk_gists: list[dict]  # All chunk summaries
    failures: list[dict]  # Failed chunks for manual review
    turns: int  # Total RLM turns
    token_usage: dict  # {root: N, worker: M, total: N+M}

# Failure list format:
failures = [
    {
        "chunk_idx": 23,
        "error": "timeout after 30s",
        "content_preview": "Invoice #INV-2024-123 | Date: 2024-01-15 | Vendor: Ac..."
    },
    {
        "chunk_idx": 45,
        "error": "malformed YAML output - unclosed bracket",
        "content_preview": "line_items: ["
    }
]
```

### Error Categories

| Category | Behavior | Retry |
|----------|----------|-------|
| **Timeout** | Worker LM took too long | Yes, once |
| **Malformed YAML** | Worker returned invalid YAML | Yes, with fix instruction |
| **API Rate Limit** | Rate limited | Yes, with backoff |
| **Network Error** | Connection failed | Yes, once |
| **Vision Unavailable** | Non-vision model on images | Fail fast (clear error) |
| **Invalid Schema** | User JSON Schema is invalid | Fail before processing |

---

## Project Structure

```
rlm-implementation/
├── rlm/
│   ├── __init__.py
│   ├── config.py              # RLMConfig (dual LM, routing)
│   ├── rlm.py                 # Core RecursiveLanguageModel
│   ├── repl.py                # REPLState for state management
│   ├── tools.py               # DSPy tools: execute, sub_lm, peek, final
│   ├── signatures.py          # DSPy signatures (Root/Worker)
│   │                           # SchemaExtractionSignature
│   │
│   ├── extract/               # Schema extraction module
│   │   ├── __init__.py
│   │   ├── extractor.py       # RLMExtractor main orchestrator
│   │   ├── schema.py          # SchemaConverter (JSON↔YAML)
│   │   ├── chunker.py         # Chunker (text/images)
│   │   └── processor.py       # ChunkProcessor with retry logic
│   │
│   └── safety.py              # Code validation, sandboxing
│
├── tests/
│   ├── __init__.py
│   ├── test_rlm.py
│   ├── test_tools.py
│   ├── extract/               # Extraction tests
│   │   ├── test_schema.py
│   │   ├── test_chunker.py
│   │   ├── test_processor.py
│   │   └── test_extractor.py
│   └── integration/
│       └── test_long_extraction.py
│
├── examples/
│   ├── needle_in_haystack.py   # Original RLM example
│   ├── extract/
│   │   ├── invoice.py          # Extract from invoice text
│   │   ├── contract.py         # Extract from legal contract
│   │   └── images.py           # Extract from document images
│   └── demo.py                 # Simple demo
│
├── docs/
│   ├── architecture.md         # Existing RLM architecture
│   ├── code_implementation.md  # Existing implementation plan
│   ├── extraction-design.md    # This document
│   └── api.md                  # API reference (to be created)
│
├── pyproject.toml              # Project configuration
├── CLAUDE.md                   # Claude Code instructions
└── README.md                   # Project overview
```

---

## API Reference

### RLMExtractor

```python
from rlm.extract import RLMExtractor, RLMConfig

# Configure
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
        chunk_size=2000,
    summary_level="standard",
    max_parallel_workers=5,
)

# Initialize
extractor = RLMExtractor(config)

# Extract from text
result = extractor.extract(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string"},
            "amount": {"type": "number"}
        }
    },
    document=open("document.txt").read(),
    task="Extract invoice information"
)

# Extract from images
result = extractor.extract(
    json_schema=schema,
    document=[image1, image2, image3],  # PIL Images
)

# Access results
print(result.data)          # Extracted JSON
print(result.failures)      # Any failures
print(result.turns)         # RLM turns used
print(result.token_usage)   # Token consumption
```

### ExtractionResult

```python
@dataclass
class ExtractionResult:
    """Result of RLM schema extraction."""

    data: dict
    """Final extracted data matching the input JSON Schema."""

    chunk_gists: list[dict]
    """Summary of each chunk: [{idx, gist, confidence, fields_found}]"""

    failures: list[dict]
    """Failed chunks: [{chunk_idx, error, content_preview}]"""

    turns: int
    """Total number of RLM orchestration turns."""

    token_usage: dict
    """Token consumption: {root: int, worker: int, total: int}"""

    def is_complete(self) -> bool:
        """True if extraction has no failures."""

    def get_failure_rate(self) -> float:
        """Percentage of chunks that failed."""
```

### SchemaConverter (Advanced)

```python
from rlm.extract.schema import SchemaConverter

converter = SchemaConverter()

# Convert JSON Schema to YAML (usually automatic)
yaml_schema = converter.json_to_yaml_chunks(json_schema)

# Convert extracted YAML back to JSON
result_json = converter.yaml_to_json(extracted_yaml, original_schema)
```

---

## Usage Examples

### Example 1: Invoice Extraction (Text)

```python
from rlm.extract import RLMExtractor, RLMConfig

config = RLMConfig(
    root_model="anthropic/claude-sonnet-4",
    worker_text_model="anthropic/claude-haiku-4",
    chunk_size=2000,
)

extractor = RLMExtractor(config)

invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "pattern": r"INV-\d{4}-\d{4}"},
        "date": {"type": "string", "format": "date"},
        "vendor": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"}
            }
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"},
                    "total": {"type": "number"}
                }
            }
        },
        "total_amount": {"type": "number"}
    },
    "required": ["invoice_number", "date", "total_amount"]
}

# Large invoice document (could be millions of characters)
document = open("large_invoice.txt").read()

result = extractor.extract(
    json_schema=invoice_schema,
    document=document
)

print(result.data)
# {
#     "invoice_number": "INV-2024-1234",
#     "date": "2024-01-15",
#     "vendor": {"name": "Acme Corp", "address": "123 Main St"},
#     "line_items": [...],
#     "total_amount": 12500.00
# }

if result.failures:
    print(f"Failed chunks: {len(result.failures)}")
```

### Example 2: Contract Extraction (Images)

```python
from PIL import Image
from rlm.extract import RLMExtractor, RLMConfig

config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
    )

extractor = RLMExtractor(config)

# Load contract pages as images
pages = [
    Image.open(f"contract_page_{i}.png")
    for i in range(1, 51)  # 50 pages
]

contract_schema = {
    "type": "object",
    "properties": {
        "contract_type": {"type": "string"},
        "parties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"}
                }
            }
        },
        "effective_date": {"type": "string"},
        "expiration_date": {"type": "string"},
        "key_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    }
}

result = extractor.extract(
    json_schema=contract_schema,
    document=pages
)

print(result.data)
```

### Example 3: Custom Task with Minimal Summary

```python
# Configure for minimal summaries (faster, less detail)
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_text_model="openai/gpt-4o-mini",
    summary_level="minimal",  # Less detailed gists
    max_parallel_workers=10,  # More parallelism
)

extractor = RLMExtractor(config)

result = extractor.extract(
    json_schema=schema,
    document=document,
    task="Extract all email addresses and phone numbers from this document"
)
```

---

## Implementation Phases

### Phase 1: Foundation (Core RLM)
- [ ] Implement `RLMConfig` with dual LM setup
- [ ] Create `REPLState` for state management
- [ ] Implement basic `RecursiveLanguageModel` with DSPy
- [ ] Add core tools: execute, sub_lm, peek, final
- [ ] Create `RLMSignature` (Root LM orchestration)

### Phase 2: Schema Conversion
- [ ] Implement `SchemaConverter` (JSON Schema → YAML)
- [ ] Implement YAML → JSON reconstruction
- [ ] Add schema validation
- [ ] Handle nested objects and arrays
- [ ] Add type conversion (strings to numbers/booleans)

### Phase 3: Chunking
- [ ] Implement `Chunker.chunk_text()` (nearest space)
- [ ] Implement `Chunker.encode_images()` (base64)
- [ ] Add chunk metadata (idx, start, end)
- [ ] Configurable chunk size

### Phase 4: Worker Extraction
- [ ] Implement `WorkerExtractionSignature`
- [ ] Create `ChunkProcessor` with retry logic
- [ ] Implement parallel worker pool
- [ ] Add gist extraction with configurable levels
- [ ] Handle text vs vision worker routing

### Phase 5: Root Orchestration
- [ ] Implement `RootExtractionSignature`
- [ ] Create `RLMExtractor` main loop
- [ ] Parallel first pass implementation
- [ ] Sequential re-extraction with targeted prompts
- [ ] Trajectory formatting and compaction

### Phase 6: Error Handling
- [ ] Implement retry strategy
- [ ] Add failure tracking and logging
- [ ] Handle all error categories
- [ ] Create `ExtractionResult` dataclass

### Phase 7: Testing
- [ ] Unit tests for SchemaConverter
- [ ] Unit tests for Chunker
- [ ] Unit tests for ChunkProcessor
- [ ] Integration tests for text extraction
- [ ] Integration tests for image extraction
- [ ] Benchmark tests for large documents

### Phase 8: Examples & Documentation
- [ ] Invoice extraction example
- [ ] Contract extraction example
- [ ] Image extraction example
- [ ] API reference documentation
- [ ] README update

---

## Configuration Reference

### RLMConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_model` | `str` | *required* | Model for orchestration (text-based) |
| `worker_text_model` | `str` | *required* | Model for text extraction |
| `chunk_size` | `int` | `2000` | Characters per text chunk |
| `summary_level` | `"minimal" \| "standard" \| "verbose"` | `"standard"` | Detail level of chunk gists |
| `parallel_first_pass` | `bool` | `True` | Parallel processing for initial extraction |
| `parallel_retry` | `bool` | `False` | Parallel processing for re-extraction |
| `max_parallel_workers` | `int` | `5` | Maximum concurrent API calls |
| `max_turns` | `int` | `20` | Maximum RLM orchestration turns |
| `code_execution_timeout` | `int` | `30` | Seconds before code execution timeout |
| `api_key` | `str \| None` | `None` | API key (or use env variable) |

### Model Recommendations

| Use Case | Root Model | Worker Text | Worker Vision |
|----------|------------|-------------|---------------|
| **Balanced** | `openai/gpt-4o` | `openai/gpt-4o-mini` | `openai/gpt-4o` |
| **Cost-optimized** | `anthropic/claude-haiku-4` | `anthropic/claude-haiku-4` | `openai/gpt-4o-mini` |
| **Quality-focused** | `anthropic/claude-sonnet-4` | `openai/gpt-4o` | `anthropic/claude-sonnet-4` |
| **Local-only** | `ollama/llama3.1` | `ollama/llama3.1` | `llava` |

### Summary Levels

| Level | Gist Format | Use Case |
|-------|-------------|----------|
| `minimal` | 1 sentence, field names only | High-throughput, familiar documents |
| `standard` | 2-3 sentences, key entities | General purpose (default) |
| `verbose` | Full paragraph, all details | Complex documents, unfamiliar formats |

---

## Open Questions

1. **DSPy Optimization**: Should we use DSPy optimizers (BootstrapFewShot, MIPROv2) to improve prompts?
   - *Decision*: Post-MVP experiment

2. **YAML Chunking**: For very large schemas, should we chunk the YAML itself?
   - *Current design*: No - workers can handle full YAML schema

3. **Schema Validation**: Should we validate the user's JSON Schema before processing?
   - *Decision*: Yes, fail fast with clear error

4. **Partial Results**: Should we return partial results when failures occur?
   - *Decision*: Yes, always return what we have + failure list

5. **Streaming**: Should we support streaming results as chunks complete?
   - *Decision*: Post-MVP feature

---

## Next Steps

1. ✅ Design document complete
2. ⏭️ Set up project structure (`pyproject.toml`, directories)
3. ⏭️ Implement Phase 1: Core RLM foundation
4. ⏭️ Implement Phase 2-3: Schema conversion and chunking
5. ⏭️ Implement Phase 4-5: Worker and Root orchestration
6. ⏭️ Implement Phase 6: Error handling
7. ⏭️ Implement Phase 7-8: Testing and examples

---

**Document Version**: 1.0
**Last Updated**: 2025-01-10
