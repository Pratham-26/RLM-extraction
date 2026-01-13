# RLM Implementation Documentation

Welcome to the documentation for the Recursive Language Model (RLM) implementation and schema extraction system.

---

## Quick Start

### Setup (using uv)

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Documentation Guide

1. **New to RLM?** Start with [Research Overview](#research)
2. **Understanding the system?** See [Architecture](#architecture)
3. **Building it?** Check [Implementation Plans](#implementation-plans)
4. **Using extraction?** Read [Extraction System](#extraction-system)

---

## Research

Original papers, study notes, and background materials on Recursive Language Models.

| Document | Description |
|----------|-------------|
| [RLM Study](research/RLM_Study.md) | Comprehensive 687-line study covering core concepts, evaluation methodology, experimental results, and implementation guide |
| [Original Paper (PDF)](research/2512.24601v1.pdf) | The research paper that introduced RLMs |
| [Original Paper (HTML)](research/html/) | Interactive HTML version of the paper |

**Key Concepts:**
- RLM treats long prompts as external environment (INPUT variable)
- Dual-model architecture: Root LM (orchestrator) + Worker LM (sub-tasks)
- Code execution in sandboxed Python REPL
- Parallel processing via worker calls

---

## Architecture

System design and component specifications.

| Document | Description |
|----------|-------------|
| [RLM Architecture](architecture/architecture.md) | Detailed 549-line architecture document covering: <br> - 5-layer architecture (Config, Signature, Safety, Tools, Orchestrator) <br> - Component specifications <br> - Execution flow <br> - Safety mechanisms <br> - Performance characteristics |

**Architecture Layers:**
1. **Configuration Layer** - Model provider abstraction (OpenAI, Anthropic, OpenRouter)
2. **Signature Layer** - DSPy input/output contracts
3. **Safety Layer** - Sandboxed Python REPL
4. **Tools Layer** - Eight tools for environment interaction
5. **Orchestrator Layer** - Main RLM coordination loop

---

## Implementation Plans

Detailed plans for implementing the RLM system.

| Document | Description |
|----------|-------------|
| [DSPy Implementation Plan](plans/code_implementation.md) | Complete implementation specifications using DSPy framework <br> - Dual-model strategy <br> - Component-by-component code specs <br> - 5 implementation phases <br> - Usage patterns |

**Implementation Phases:**
1. Project setup (pyproject.toml, dependencies)
2. Core RLM module (RLMConfig, REPLState, signatures)
3. Tools integration (execute, sub_lm, peek, final)
4. Testing & optimization
5. Examples

---

## Extraction System

Schema-based information extraction using RLM concepts.

| Document | Description |
|----------|-------------|
| [Extraction Design](extraction/extraction-design.md) | Complete design for schema-based extraction <br> - Architecture overview <br> - Component specifications <br> - Data flows (text & image modes) <br> - API reference <br> - Usage examples |

**Extraction Features:**
- Extract from arbitrarily large documents (text or images)
- JSON Schema input → JSON output (YAML is internal)
- Parallel first pass + sequential re-examination
- Configurable chunking and parallelism
- Transparent failure tracking

---

## Project Status

| Component | Status |
|-----------|--------|
| Documentation | ✅ Complete |
| RLM Implementation | ✅ Complete |
| Extraction System | ✅ Complete |
| Tests | ✅ Complete |
| Examples | ✅ Complete |

---

## Related Files

- [`CLAUDE.md](../CLAUDE.md) - Instructions for Claude Code
- [`README.md`](../README.md) - Project overview
- [`pyproject.toml`](../pyproject.toml) - Project configuration
