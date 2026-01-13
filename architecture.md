# RLM Architecture

## Conceptual Overview

Recursive Language Models (RLM) enable Large Language Models to process **arbitrarily long inputs** by treating the prompt as an external environment rather than trying to fit everything into the context window.

### The Core Problem

LLMs have finite context windows. When documents exceed this limit:
- Information gets truncated
- Retrieval becomes complex (requires RAG systems)
- Cross-document relationships are lost
- Costs scale with context size

### The RLM Solution

Instead of feeding the entire document to the LLM, RLM provides a **Python REPL environment** where:
- The document exists as a variable called `INPUT`
- The LLM writes and executes code to explore the document
- Processing scales algorithmically, not by context size

---

## Dual-Model Architecture

RLM uses two distinct model roles, each optimized for their purpose:

### Root LM (Orchestrator)

**Purpose**: Strategic planning and coordination

**Capabilities**:
- Analyzes what information is needed
- Writes code to process the input
- Coordinates worker models
- Aggregates and synthesizes results

**Key Characteristic**: Never sees the full document directly. Only sees:
- Task description
- Execution results
- Chunk summaries
- Worker outputs

### Worker LM (Executor)

**Purpose**: Tactical processing of assigned chunks

**Capabilities**:
- Processes specific document segments
- Extracts information according to schema
- Works in parallel with other workers
- Returns structured results

**Key Characteristic**: Sees only the chunk it's assigned plus the extraction schema.

### Why This Separation Matters

| Aspect | Root LM | Worker LM |
|--------|---------|-----------|
| **Model size** | Larger, more capable | Smaller, faster |
| **Context** | Results and summaries | Raw document chunks |
| **Parallelism** | Sequential (orchestration) | Parallel (processing) |
| **Cost driver** | Turn count | Chunk count |
| **Optimization** | Minimize turns | Maximize throughput |

---

## System Architecture

### High-Level Structure

```
User Request
     |
     v
+-------------------+
|  RLM Orchestrator |  (Root LM - Strategic)
+-------------------+
     |
     | Coordinates
     v
+-------------------+
|  Processing Layer |
+-------------------+
     |
     +-- Code Execution --> Python REPL (INPUT variable)
     |
     +-- Worker Calls --> Parallel Worker LMs
     |
     +-- Context Access --> Chunk retrieval
     |
     +-- Result Synthesis --> Final answer
```

### Information Flow

1. **Input Phase**: Document stored as `INPUT` variable in REPL
2. **Planning Phase**: Root LM analyzes task and writes approach code
3. **Execution Phase**: Code executes, workers process chunks in parallel
4. **Aggregation Phase**: Root LM synthesizes results into final answer
5. **Output Phase**: Structured data returned to user

---

## Key Design Principles

### 1. Environment as External State

The document is **not** part of the prompt. It exists in the REPL as the `INPUT` variable. The LLM interacts with it through code execution.

**Why**: Decouples input size from context window limits.

### 2. Code as Interface

The LLM writes Python code to:
- Search for patterns (regex, string matching)
- Filter and transform data
- Coordinate parallel processing
- Aggregate results

**Why**: LLMs are excellent at writing code. Code scales better than natural language processing.

### 3. Trajectory as Context

Instead of feeding the entire document, the Root LM sees:
- History of actions taken
- Results from those actions
- Current state (variables, findings)

**Why**: Provides relevant context without overwhelming the LLM.

### 4. Parallel Worker Execution

Multiple workers process chunks simultaneously:
- Each worker sees only its assigned chunk
- Workers operate independently
- Results merge back to Root LM

**Why**: Reduces latency and increases throughput.

---

## Component Responsibilities

### Configuration Layer

**Purpose**: Abstract model provider differences

**Responsibilities**:
- Manage API keys and endpoints
- Create LM instances (Root and Worker)
- Handle provider-specific quirks
- Support multiple providers (OpenAI, Anthropic, OpenRouter)

### Signature Layer

**Purpose**: Define LM interaction contracts

**Responsibilities**:
- Specify input/output formats declaratively
- Enable type-safe LM calls
- Support prompt optimization

**Key Signatures**:
- **Root Decision**: Task → Next action
- **Context Condensation**: Verbose user input → Concise guidance
- **Worker Extraction**: Schema + Chunk → Extracted entities
- **Value Extraction**: Entity contexts → Final values

### Safety Layer

**Purpose**: Sandbox code execution

**Responsibilities**:
- Validate code before execution
- Block dangerous operations (file I/O, network, system calls)
- Provide safe built-ins and modules
- Enforce execution timeouts

### REPL State Layer

**Purpose**: Maintain persistent state across turns

**Responsibilities**:
- Store INPUT (the document)
- Track variables created during execution
- Maintain trajectory history
- Manage chunk processing status

### Orchestration Layer

**Purpose**: Coordinate the entire extraction process

**Responsibilities**:
- Convert JSON Schema to internal YAML format
- Chunk documents for processing
- Run parallel first-pass extraction
- Orchestrate sequential re-extraction
- Aggregate and format results

---

## Data Flow Patterns

### Pattern 1: Needle in Haystack

**Goal**: Find specific information in a massive document

**Flow**:
1. Root LM writes code to search for patterns
2. Code executes and returns matching portions
3. Root LM returns the needle

**Parallelism**: Low (search is sequential)

### Pattern 2: Divide and Conquer

**Goal**: Process entire input (e.g., extract all fields)

**Flow**:
1. Root LM chunks the input
2. Workers process chunks in parallel
3. Root LM aggregates results

**Parallelism**: High (all chunks independent)

### Pattern 3: Multi-Hop Question Answering

**Goal**: Connect information across locations

**Flow**:
1. Root LM finds first evidence
2. Root LM searches for related evidence
3. Root LM synthesizes answer

**Parallelism**: Medium (dependent searches)

---

## Scaling Characteristics

### Input Size

- **Scales**: Linearly with document length
- **Bottleneck**: Code execution speed, not LM context
- **Limit**: None (theoretical)

### Cost

- **Root LM**: Proportional to turn count (typically < 10)
- **Worker LM**: Proportional to chunk count (parallelized)
- **Optimization**: Use cheaper models for workers

### Latency

- **Per turn**: 2-5 seconds (Root LM call)
- **Parallel chunks**: Amortized across workers
- **Total**: Turns × (Root LM time + Worker time / parallelism)

---

## Safety Considerations

### Multi-Layer Protection

1. **Static Analysis**: Parse code into AST, check for blocked patterns
2. **Runtime Sandboxing**: Restricted globals, no dangerous modules
3. **Timeout Protection**: Code execution wrapped with timeout
4. **Signal Handling**: SIGALRM for timeout enforcement (Unix)

### Allowed Operations

- String manipulation and regex
- Data structure operations (lists, dicts, sets)
- Mathematical computations
- JSON parsing and serialization
- Iteration and comprehension

### Blocked Operations

- File I/O (open, read, write)
- Network calls (socket, requests, urllib)
- System manipulation (os, subprocess, sys)
- Dynamic code execution (exec, eval, __import__)

---

## State Management

### Trajectory

The trajectory is a history of actions and results:
- **thought**: Root LM's reasoning
- **action**: Action taken (code, sub_lm, peek, final)
- **result**: Output from executing the action

### Trajectory Compaction

As turns increase, trajectory grows. RLM implements a sliding window:
- Keep recent entries (what just happened)
- Discard ancient entries (what happened long ago)
- Preserve important variables in REPL state

**Why**: Recent context is most relevant for deciding next action. REPL state preserves any variables created early.

---

## Usage Patterns

### When to Use RLM

- **Documents exceed context window** (primary use case)
- **Need structured extraction** from unstructured text
- **Cost-sensitive processing** of large inputs
- **Complex, multi-step reasoning** over documents

### When NOT to Use RLM

- **Small documents** that fit in context
- **Simple queries** without cross-document relationships
- **Real-time streaming** requirements (current design is batch)
- **Need for fine-grained control** over LM prompts

---

## Model Provider Strategy

### DSPy LM Abstraction

RLM uses DSPy's unified LM interface:
- `basic_request`: Make API call
- `__call__`: Get completion
- `copy`: Create LM with different parameters

### Provider Switching

Factory pattern enables easy switching:
```python
# Conceptual (not code)
config = create_config(
    provider="openrouter",  # or "openai", "anthropic"
    root_model="claude-sonnet-4",
    worker_model="claude-haiku-4"
)
```

### Multi-Provider Support

- **OpenRouter**: 100+ models, unified API
- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet, Claude Haiku
- **Local**: Ollama, vLLM (via OpenRouter-compatible endpoints)

---

## Summary

RLM architecture enables unlimited input processing through:

1. **External Environment**: Document as INPUT variable, not in prompt
2. **Dual-Model Design**: Root (orchestrator) + Workers (executors)
3. **Code Interface**: LLM writes Python to explore and process
4. **Parallel Execution**: Workers process chunks simultaneously
5. **Trajectory Management**: History of actions with compaction
6. **Safety Layers**: Sandboxed code execution
7. **Provider Abstraction**: Easy model switching

The result is a system that scales to arbitrarily large documents while maintaining cost efficiency and extraction quality.
