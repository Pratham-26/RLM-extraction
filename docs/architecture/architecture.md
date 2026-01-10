# RLM Architecture and Working

This document provides a detailed explanation of the Recursive Language Model (RLM) architecture and how it processes arbitrarily long contexts beyond the limitations of native LLM context windows.

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Core Problem](#core-problem)
3. [RLM Solution](#rlm-solution)
4. [System Architecture](#system-architecture)
5. [Component Deep Dive](#component-deep-dive)
6. [Execution Flow](#execution-flow)
7. [Data Structures](#data-structures)
8. [Safety Mechanisms](#safety-mechanisms)
9. [Trajectory Management](#trajectory-management)
10. [Model Provider Support](#model-provider-support)

---

## Conceptual Overview

### What is RLM?

Recursive Language Models (RLM) is a paradigm that enables Large Language Models to process inputs of unlimited length by treating the prompt as an external environment rather than trying to fit everything into the context window.

The key insight is that LLMs are excellent at writing code to solve problems. Instead of forcing a massive document into the context window, RLM gives the LLM a Python REPL environment where the long document exists as a variable called INPUT. The LLM can then write and execute code to inspect, analyze, and process this input in chunks.

### Why This Matters

Modern LLMs have context windows ranging from 128K to 2M tokens. While impressive, these are still finite limits. For real-world use cases like:

- Analyzing multi-million-line codebases
- Processing entire legal document libraries
- Searching through massive log files
- Cross-referencing large datasets

The context window will eventually be exceeded. RLM solves this by not trying to fit everything into context at once.

---

## Core Problem

### The Context Window Barrier

When you feed a document that exceeds the context window to a traditional LLM approach, you face several problems:

1. **Truncation**: Information at the end gets cut off
2. **Retrieval Complexity**: You need sophisticated RAG systems to find relevant chunks
3. **Loss of Context**: The LLM cannot see relationships between distant parts
4. **Cost**: Processing huge contexts is expensive even when supported

### The RLM Approach

RLM changes the paradigm:

- Instead of: "Here's the entire document, answer questions"
- Use: "Here's a Python REPL with the document as INPUT. Write code to find the answer."

This shifts the burden from the LLM's context window to algorithmic processing, which scales infinitely.

---

## RLM Solution

### Dual-Model Architecture

RLM uses two types of models strategically:

**Root LM (Orchestrator)**
- A larger, more capable model (e.g., GPT-4, Claude Sonnet)
- Responsible for: strategy, planning, code generation, coordination
- Sees: the task, trajectory history, and execution results
- Does NOT see: the full INPUT directly

**Worker LM (Sub-tasks)**
- A smaller, faster model (e.g., GPT-4o-mini, Claude Haiku, Llama)
- Responsible for: processing chunks, extraction, verification
- Sees: specific chunks of context delegated by root
- Benefits: cheaper, faster, can run in parallel

This separation provides both cost optimization and scalability.

---

## System Architecture

### High-Level Structure

The RLM system consists of five main components:

1. **Configuration Layer** - Manages model providers and settings
2. **Signature Layer** - Defines input/output contracts using DSPy
3. **Safety Layer** - Sandboxed Python REPL for code execution
4. **Tools Layer** - Eight tools for interacting with the environment
5. **Orchestrator Layer** - Main RLM loop that coordinates everything

### Component Diagram

```
User Request
     |
     v
+-------------------+
|  RLM Orchestrator |<----+ Configuration (Root LM, Worker LM)
+-------------------+     |
     |                     |
     | uses               | provides
     v                     |
+-------------------+     |
|   DSPy Signatures |     |
+-------------------+     |
     |                     |
     v                     |
+-------------------+     |
|  Toolkit (8 tools)|-----+
+-------------------+
     |
     +-- python_execute --> REPLState (INPUT variable)
     |
     +-- sub_lm_call ----> Worker LM (parallel processing)
     |
     +-- peek_context --> Direct INPUT access
     |
     +-- get_variable --> REPL variable access
     |
     +-- final_answer --> Return result to user
```

---

## Component Deep Dive

### 1. Configuration Layer

The configuration layer abstracts away the differences between model providers. It supports:

- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet, Claude Haiku
- **OpenRouter**: 100+ models via unified API

Each provider is implemented as a class that:
- Manages API keys and endpoints
- Creates DSPy LM instances
- Handles provider-specific quirks (headers, response formats)
- Supports dual-model configuration (root + worker)

The OpenRouter integration is particularly interesting because it uses a custom LM class that inherits from DSPy's base LM class and implements the required methods for API communication.

### 2. Signature Layer

DSPy signatures define the "contracts" for LM interactions. Instead of writing brittle prompts, we define input and output fields declaratively.

**Root LM Signature**
The root LM receives:
- The task description
- The trajectory (history of actions taken)
- Context information (length, previews, last results)

And produces:
- A thought process
- The next action to take
- Arguments for that action

**Worker LM Signature**
Worker LMs receive:
- A specific task or prompt
- A chunk of context to process

And produce:
- A direct result

This separation allows the root to think strategically while workers handle tactical sub-tasks.

### 3. Safety Layer

The safety layer provides a sandboxed Python REPL environment. This is critical because the root LM will be executing arbitrary code that it writes itself.

**REPL State Management**
The REPL maintains:
- INPUT variable containing the long context
- Persistent variables across executions
- An LM() function for recursive calls
- Safe execution environment

**Code Validation**
Before executing any code, the validator checks for:

Blocked Modules:
- os, subprocess, sys - Could manipulate the host system
- socket, urllib, requests - Could make network calls
- Various other dangerous modules

Blocked Built-ins:
- open - Could read/write arbitrary files
- exec, eval, __import__ - Could execute arbitrary code
- compile - Could compile malicious code

Dangerous Patterns:
- __globals__, __class__, __code__ - Could access unsafe objects
- Various introspection patterns

**Safe Execution**
The validator also creates a safe globals dictionary that only includes:
- Safe built-in functions (len, print, str, etc.)
- Safe modules (math, re, json, collections, etc.)
- The INPUT variable
- The LM() function

### 4. Tools Layer

The toolkit provides eight tools that the root LM can use to interact with its environment:

**Python Execute Tool**
Executes Python code in the REPL. This is the primary tool for processing INPUT. The root LM uses this to write analysis code, filtering logic, chunking operations, etc.

**Sub-LM Call Tool**
Makes recursive calls to worker LMs. This enables parallel processing. The root can delegate chunks to multiple workers that process simultaneously.

**Peek Context Tool**
Provides a quick way to view portions of INPUT without writing code. Useful for initial exploration.

**Get Variable Tool**
Retrieves variables created by previous code executions. This allows the LM to build state over multiple turns.

**Get Context Info Tool**
Returns metadata about INPUT (length, preview) without code execution.

**Final Answer Tool**
Signals task completion. When this tool is used, the RLM loop terminates and returns the answer.

**Verify Answer Tool**
Allows the LM to self-verify its answers against the context. Useful for catching hallucinations.

**Aggregate Results Tool**
Combines outputs from multiple worker LM calls into a coherent final answer.

### 5. Orchestrator Layer

The orchestrator is the heart of RLM. It implements the main loop that:

1. Initializes the REPL and toolkit
2. Formats the trajectory for the root LM
3. Calls the root LM to get the next action
4. Executes that action using the appropriate tool
5. Records the result in the trajectory
6. Compacts the trajectory if it grows too large
7. Repeats until final answer or max turns reached

The orchestrator also implements trajectory compaction, which is crucial for long-running tasks.

---

## Execution Flow

### Initialization Phase

When a user calls RLM with a task and input context:

1. Configuration is loaded and DSPy is configured with the root LM
2. REPL state is created with INPUT set to the input context
3. The toolkit is initialized with all tools bound to the REPL
4. The sub-LM tool is created with the worker LM
5. An empty trajectory is started

### Main Loop Phase

For each turn (up to max_turns):

**Step 1: Trajectory Compaction**
The orchestrator checks if the formatted trajectory exceeds max_trajectory_length. If so, it removes the oldest entries while keeping at least min_trajectory_entries. This prevents context overflow.

**Step 2: Root LM Call**
The root LM receives:
- The task
- The formatted trajectory
- Context information

The root LM outputs:
- A thought process
- The next action to take
- Arguments for that action

**Step 3: Action Execution**
Based on the action type, the orchestrator routes to the appropriate tool:

For "code" actions:
- The code is validated for safety
- If safe, executed in the REPL
- Output is captured and returned

For "sub_lm" actions:
- The worker LM is called with the prompt and optional context
- The result is returned
- Multiple calls can be made in parallel

For "peek" actions:
- A substring of INPUT is returned directly

For "final" actions:
- The answer is extracted and returned to the user
- The loop terminates

**Step 4: Trajectory Update**
The action, thought, and result are appended to the trajectory. This creates a history that the root LM can reference in future turns.

### Termination

The loop ends when:
- The root LM uses the "final" action, OR
- max_turns is reached

The result includes:
- The final answer (or error message)
- The complete trajectory of actions taken
- Number of turns completed

---

## Data Structures

### Trajectory

The trajectory is the history of the root LM's reasoning and actions. Each entry contains:

- thought: The root LM's reasoning process
- action: The action taken (code, sub_lm, peek, etc.)
- result: The output from executing that action

The trajectory grows with each turn and provides context to the root LM about what has been tried and what the results were.

### REPL State

The REPL maintains:

- input_context: The original INPUT string
- variables: Dictionary of variables created during execution
- execution_count: Number of code executions performed
- globals: Safe globals dictionary for code execution
- locals: Local variables from the last execution

This state persists across all turns in a single RLM invocation.

### Configuration

The configuration dataclass contains:

- root_model: Model identifier for the orchestrator
- worker_model: Model identifier for sub-tasks
- max_turns: Maximum loop iterations
- max_parallel_workers: Maximum parallel sub-LM calls
- code_execution_timeout: Timeout for code execution
- max_trajectory_length: Threshold for compaction
- min_trajectory_entries: Minimum entries to preserve during compaction

---

## Safety Mechanisms

### Multi-Layer Protection

The RLM implementation includes multiple layers of safety:

**Layer 1: Static Analysis**
Before execution, code is parsed into an AST and checked for:
- Import statements targeting blocked modules
- Calls to blocked built-in functions
- Dangerous code patterns

**Layer 2: Runtime Sandboxing**
Even if malicious code passes static checks, the runtime environment:
- Provides only safe built-in functions
- Blocks access to dangerous modules
- Prevents file I/O and network operations

**Layer 3: Timeout Protection**
Code execution is wrapped in a timeout to prevent infinite loops or hanging operations.

**Layer 4: Signal Handling**
On Unix systems, SIGALRM is used to enforce timeouts. On Windows, this gracefully degrades.

### What CAN Be Done

Safe operations include:
- String manipulation and regex
- Data structure operations (lists, dicts, sets)
- Mathematical computations
- JSON parsing and serialization
- Iteration and comprehension
- Variable assignment and retrieval

These operations are sufficient for most text analysis and information extraction tasks.

---

## Trajectory Management

### The Problem of Growing Context

As the RLM loop progresses, the trajectory grows. Each turn adds the root LM's thought, action, and result. For long-running tasks, this can exceed the root LM's context window.

### Compaction Algorithm

The compaction algorithm implements a sliding window:

1. Calculate the character length of the formatted trajectory
2. If below threshold, no compaction needed
3. If above threshold:
   - Remove the oldest trajectory entry
   - Recalculate length
   - Repeat until below threshold OR only min_entries remain
4. If still too large, raise an error

### Why This Works

The sliding window preserves:
- Recent context (what the LM just did)
- Current state (variables, last results)

And discards:
- Ancient history (what the LM did 20 turns ago)

This is effective because the root LM primarily needs to know what happened recently to continue effectively. The REPL state preserves any important variables that were created early on.

---

## Model Provider Support

### DSPy LM Abstraction

DSPy provides a unified LM interface that RLM leverages. The key methods are:

- basic_request: Makes a single API call
- __call__: Convenience method for getting completions
- copy: Creates a copy with updated parameters

### OpenRouter Integration

OpenRouter provides access to 100+ models through a unified API. The custom OpenRouterLM class:

- Inherits from DSPy's LM base class
- Implements required methods (basic_request, __call__, copy)
- Adds OpenRouter-specific headers (HTTP-Referer, X-Title)
- Handles the OpenRouter response format

The headers are important for OpenRouter's leaderboard attribution system, which tracks usage by application.

### Provider Switching

The factory pattern enables easy provider switching:

- Import create_config function
- Specify provider ("openai", "anthropic", "openrouter")
- Optionally override default models
- Get a fully configured RLMConfig instance

This makes it easy to:
- Switch between providers based on cost/availability
- A/B test different models
- Use specialized models for specific tasks

---

## Usage Patterns

### Needle in Haystack

For finding specific information in a massive document:

1. Root LM uses "peek" to understand INPUT structure
2. Root LM writes code to search for patterns (regex, keyword matching)
3. Code executes and returns matching portions
4. Root LM uses "final" to return the needle

### Divide and Conquer

For processing that requires analyzing the entire input:

1. Root LM writes code to chunk INPUT into pieces
2. Root LM uses "sub_lm" to process chunks in parallel
3. Workers extract relevant information from their chunks
4. Root LM uses "aggregate" to combine results
5. Root LM uses "final" to return the synthesized answer

### Multi-Hop Question Answering

For complex questions requiring connecting information across locations:

1. Root LM uses "code" to find first piece of evidence
2. Root LM uses "code" to find related evidence
3. Root LM uses "verify" to check if evidence supports the answer
4. Root LM uses "final" to return the reasoned answer

---

## Performance Characteristics

### Scalability

RLM scales linearly with input size for most operations:
- String operations: O(n) where n is input length
- Regex search: O(n) average case
- Chunking: O(n) to divide input

Parallel worker calls provide:
- Near-linear speedup for independent sub-tasks
- Configurable via max_parallel_workers
- Typical 5-10x speedup vs sequential processing

### Cost Optimization

The dual-model architecture reduces costs:
- Root LM (expensive) sees only summaries and results
- Worker LM (cheap) processes raw context chunks
- Typical cost reduction: 60-80% vs single-model approach

### Latency

Turn-based processing adds latency:
- Each turn requires a root LM call
- Sequential turns execute one after another
- Typical latency: 2-5 seconds per turn

Parallel sub-LM calls mitigate this:
- Multiple workers execute simultaneously
- Reduces effective latency for parallelizable tasks

---

## Summary

RLM is a system for processing arbitrarily long contexts by:

1. Treating the prompt as an external environment (INPUT variable)
2. Using a powerful root LM to orchestrate and write code
3. Executing code in a sandboxed Python REPL
4. Making parallel calls to cheaper worker models
5. Maintaining a trajectory of actions and results
6. Compacting the trajectory to prevent context overflow
7. Returning a final answer when complete

The combination of these techniques enables:
- Unlimited input length
- Cost-efficient processing
- Parallel execution
- Safe code execution
- Flexible model provider support

This architecture is implemented using DSPy for clean separation of concerns, composable components, and declarative LM programming.
