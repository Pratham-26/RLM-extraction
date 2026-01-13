# RLM Implementation Plan (DSPy-Based)

## Architecture Overview

### Why DSPy?

DSPy is the ideal framework for RLM because:
- **Declarative**: Define signatures (input/output specs) instead of brittle prompts
- **Composable**: Build complex programs from modular components
- **Built-in ReAct**: The `dspy.ReAct` pattern is very close to RLM's iterative approach
- **PythonInterpreter**: Native code execution tool (similar to RLM's REPL)
- **Multi-LM support**: Easy to configure different models for root vs. worker
- **Optimization**: Built-in optimizers can improve prompts automatically

### Dual-Model Strategy

| Component | Model Type | DSPy Configuration | Role |
|-----------|------------|-------------------|------|
| **Root LM (Orchestrator)** | Large, capable model | `dspy.LM("openai/gpt-4o")` | Plans strategy, writes code, aggregates results |
| **Worker LM (Sub-LM)** | Smaller, faster model | `dspy.LM("openai/gpt-4o-mini")` | Executes sub-tasks on context chunks |

**Rationale**: The orchestrator needs strong reasoning and coding capabilities, while workers perform simpler extraction/comparison tasks that smaller models handle well.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RLM as DSPy Module                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              RecursiveLanguageModel (dspy.Module)             │   │
│  │                                                                │   │
│  │  ┌─────────────────┐    ┌────────────────────────────────┐   │   │
│  │  │ Root Predictor  │    │   Tools (DSPy Tools)            │   │   │
│  │  │ (dspy.ChainOfThought)│   - python_execute(code)       │   │   │
│  │  │                 │    │   - sub_lm_call(prompt)        │   │   │
│  │  │ Generates:      │    │   - peek_context(start, end)   │   │   │
│  │  │ - thought       │    │   - get_context_length()       │   │   │
│  │  │ - action        │    │   - final_answer(answer)       │   │   │
│  │  │ - code (optional)│   │                                │   │   │
│  │  └─────────────────┘    └────────────────────────────────┘   │   │
│  │                                                                │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │              REPL State                              │   │   │
│  │  │  - INPUT: str (long context)                         │   │   │
│  │  │  - variables: dict (persistent state)                │   │   │
│  │  │  - trajectory: list (history of actions)             │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Worker Pool (Async)                            │    │
│  │  Uses separate LM: dspy.LM("openai/gpt-4o-mini")          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rlm-implementation/
├── rlm/
│   ├── __init__.py
│   ├── config.py                # DSPy LM configuration
│   ├── rlm.py                   # Main RecursiveLanguageModel module
│   ├── tools.py                 # DSPy Tool implementations
│   ├── signatures.py            # DSPy Signatures for RLM
│   └── safety.py                # Code execution safety wrapper
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_rlm.py
│   └── integration/
│       └── test_long_context.py
├── examples/
│   ├── needle_in_haystack.py
│   └── document_qa.py
├── pyproject.toml
└── README.md
```

---

## Component Specifications (DSPy-Based)

### 1. Configuration (`rlm/config.py`)

```python
import dspy
from typing import Literal

class RLMConfig:
    """RLM configuration using DSPy's LM system."""

    def __init__(
        self,
        root_model: str = "openai/gpt-4o",
        worker_model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        max_turns: int = 20,
        max_parallel_workers: int = 5,
        code_execution_timeout: int = 30,
    ):
        # Configure root LM (orchestrator) - stronger model
        self.root_lm = dspy.LM(
            root_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_tokens=4096,
            temperature=0.0,
        )

        # Configure worker LM (sub-calls) - faster, cheaper model
        self.worker_lm = dspy.LM(
            worker_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_tokens=2048,
            temperature=0.0,
        )

        self.max_turns = max_turns
        self.max_parallel_workers = max_parallel_workers
        self.code_execution_timeout = code_execution_timeout

    def configure_dspy(self):
        """Set up DSPy with the root LM as default."""
        dspy.configure(lm=self.root_lm, track_usage=True)
```

### 2. Signatures (`rlm/signatures.py`)

```python
import dspy

class RLMThoughtSignature(dspy.Signature):
    """Signature for a single RLM reasoning step."""

    task = dspy.InputField(desc="The task to solve")
    trajectory = dspy.InputField(desc="History of previous actions and observations")
    context_info = dspy.InputField(desc="Information about available context (length, preview)")

    thought = dspy.OutputField(desc="Reasoning about next action")
    action = dspy.OutputField(desc="Next action: 'code', 'sub_lm', 'peek', or 'final'")
    action_args = dspy.OutputField(desc="Arguments for the action (JSON or code)")

class WorkerSignature(dspy.Signature):
    """Signature for worker LM sub-calls."""

    prompt = dspy.InputField(desc="The task/prompt for the worker")
    context = dspy.InputField(desc="The context chunk to process")

    result = dspy.OutputField(desc="The processed result")
```

### 3. Tools (`rlm/tools.py`)

```python
import dspy
from dspy.primitives.tool import Tool

class PythonExecuteTool(Tool):
    """Execute Python code in the RLM REPL environment."""

    def __init__(self, repl_state: "REPLState"):
        super().__init__(
            name="python_execute",
            func=self.execute,
            desc="Execute Python code with access to INPUT variable and persistent state"
        )
        self.repl_state = repl_state

    def execute(self, code: str) -> str:
        """Execute code and return output."""
        return self.repl_state.execute(code)

class SubLMCallTool(Tool):
    """Make a recursive sub-LM call using the worker model."""

    def __init__(self, worker_lm: dspy.LM, max_parallel: int = 5):
        super().__init__(
            name="sub_lm_call",
            func=self.call,
            desc="Make a sub-LM call to process a task (uses cheaper worker model)"
        )
        self.worker_lm = worker_lm
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)

    def call(self, prompt: str, context: str | None = None) -> str:
        """Execute sub-LM call."""
        # Use worker LM for the sub-call
        with dspy.context(lm=self.worker_lm):
            worker = dspy.Predict(WorkerSignature)
            return worker(prompt=prompt, context=context).result

class PeekContextTool(Tool):
    """Peek at a portion of the INPUT without loading everything."""

    def __init__(self, repl_state: "REPLState"):
        super().__init__(
            name="peek_context",
            func=self.peek,
            desc="View a portion of INPUT: peek(start, end)"
        )
        self.repl_state = repl_state

    def peek(self, start: int, end: int) -> str:
        """Return a slice of INPUT."""
        return self.repl_state.INPUT[start:end]

class FinalAnswerTool(Tool):
    """Signal completion and return final answer."""

    def __init__(self):
        super().__init__(
            name="final_answer",
            func=self.finalize,
            desc="Mark task complete and return the final answer"
        )

    def finalize(self, answer: str) -> str:
        """Return the final answer."""
        return f"FINAL({answer})"
```

### 4. REPL State (`rlm/safety.py`)

```python
import io
import re
from contextlib import redirect_stdout

class REPLState:
    """Manages the Python REPL state for RLM."""

    BLOCKED_MODULES = {"os", "subprocess", "shutil", "sys", "socket", "urllib", "requests"}
    BLOCKED_OPERATIONS = {"open(", "exec(", "eval(", "__import__", "compile("}

    def __init__(self, input_context: str, sub_lm_tool: SubLMCallTool):
        self.INPUT = input_context
        self.variables = {}
        self.globals = {
            "INPUT": input_context,
            "LM": sub_lm_tool.call,  # Inject LM() function
        }
        self.locals = {}

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Check code for dangerous operations."""
        for module in self.BLOCKED_MODULES:
            if f"import {module}" in code:
                return False, f"Blocked module: {module}"

        for op in self.BLOCKED_OPERATIONS:
            if op in code:
                return False, f"Blocked operation: {op}"

        return True, "OK"

    def execute(self, code: str) -> str:
        """Execute Python code and return output."""
        # Validate first
        safe, msg = self.validate_code(code)
        if not safe:
            return f"Error: {msg}"

        try:
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, self.globals, self.locals)

            # Update variables from locals
            for k, v in self.locals.items():
                if not k.startswith("_"):
                    self.variables[k] = v

            result = output.getvalue()
            return result if result else "Code executed successfully (no output)"
        except Exception as e:
            return f"Error: {str(e)}"
```

### 5. Main RLM Module (`rlm/rlm.py`)

```python
import dspy
from dspy.primitives.module import Module
from concurrent.futures import ThreadPoolExecutor

class RecursiveLanguageModel(dspy.Module):
    """Main RLM module implemented as a DSPy Module."""

    def __init__(self, config: RLMConfig):
        super().__init__()
        self.config = config

        # Initialize REPL state
        self.sub_lm_tool = SubLMCallTool(config.worker_lm, config.max_parallel_workers)
        self.repl_state = None  # Set in forward()

        # Create the reasoning predictor
        self.thinker = dspy.ChainOfThought(RLMThoughtSignature)

        # Create tools list (ReAct-style)
        self.tools = [
            PythonExecuteTool(self.sub_lm_tool),  # Will bind to repl_state in forward
            self.sub_lm_tool,
            PeekContextTool(self.sub_lm_tool),   # Will bind to repl_state in forward
            FinalAnswerTool(),
        ]

    def forward(self, task: str, input_context: str) -> dspy.Prediction:
        """Execute the RLM loop."""
        # Initialize REPL state for this task
        repl = REPLState(input_context, self.sub_lm_tool)

        # Update tools with correct repl_state
        self.tools[0].repl_state = repl
        self.tools[2].repl_state = repl

        trajectory = []
        context_info = f"INPUT length: {len(input_context)} chars"

        for turn in range(self.config.max_turns):
            # Get thought and action from root LM
            result = self.thinker(
                task=task,
                trajectory=self._format_trajectory(trajectory),
                context_info=context_info,
            )

            trajectory.append({"thought": result.thought})

            # Execute action
            match result.action:
                case "code":
                    observation = repl.execute(result.action_args)
                    trajectory.append({"action": "code", "result": observation})
                    context_info = f"Last execution result: {observation[:200]}"

                case "sub_lm":
                    observation = self.sub_lm_tool.call(**result.action_args)
                    trajectory.append({"action": "sub_lm", "result": observation})

                case "peek":
                    start, end = self._parse_peek_args(result.action_args)
                    observation = repl.INPUT[start:end]
                    trajectory.append({"action": "peek", "result": observation})

                case "final":
                    # Extract answer and return
                    answer = self._extract_final_answer(result.action_args)
                    return dspy.Prediction(
                        answer=answer,
                        trajectory=trajectory,
                        usage=self.thinker.get_lm_usage()
                    )

        # Max turns reached
        return dspy.Prediction(
            answer="Max turns reached without final answer",
            trajectory=trajectory
        )

    def _format_trajectory(self, trajectory: list) -> str:
        """Format trajectory for prompt."""
        return "\n".join([
            f"Step {i}: {step.get('thought', step.get('action', ''))}"
            for i, step in enumerate(trajectory)
        ])

    def _parse_peek_args(self, args: str) -> tuple[int, int]:
        """Parse peek arguments."""
        import json
        data = json.loads(args)
        return data.get("start", 0), data.get("end", 1000)

    def _extract_final_answer(self, args: str) -> str:
        """Extract final answer from action args."""
        # Handle FINAL(answer) format or JSON
        if "FINAL(" in args:
            import re
            match = re.search(r"FINAL\((.*?)\)", args)
            return match.group(1) if match else args
        return args
```

---

## Implementation Phases

### Phase 1: Project Setup
- [ ] Create pyproject.toml with DSPy dependency
- [ ] Set up project structure
- [ ] Configure DSPy with API keys

### Phase 2: Core RLM Module
- [ ] Implement `RLMConfig` with dual LM setup
- [ ] Create `REPLState` for code execution
- [ ] Implement `RLMSignature` for reasoning steps
- [ ] Build basic `RecursiveLanguageModel` module

### Phase 3: Tools Integration
- [ ] Implement `PythonExecuteTool` with safety
- [ ] Create `SubLMCallTool` with worker LM
- [ ] Add `PeekContextTool` for context inspection
- [ ] Implement `FinalAnswerTool` for completion

### Phase 4: Testing & Optimization
- [ ] Unit tests for tools
- [ ] Integration tests with sample tasks
- [ ] Benchmark on needle-in-haystack
- [ ] Optional: Use DSPy optimizers (BootstrapFewShot, MIPROv2)

### Phase 5: Examples
- [ ] Needle-in-haystack example
- [ ] Long document QA example
- [ ] Code repository understanding example

---

## Key DSPy Patterns Used

| Pattern | Usage | Benefit |
|---------|-------|---------|
| `dspy.Module` | Base class for RLM | Composability, learnable parameters |
| `dspy.Signature` | Define input/output contracts | Type safety, clear interfaces |
| `dspy.ChainOfThought` | Root LM reasoning | Step-by-step thinking before action |
| `dspy.Tool` | ReAct-style tool calling | Structured action selection |
| `dspy.context(lm=...)` | Switch between LMs | Dual-model orchestration |
| `dspy.configure()` | Global LM setup | Centralized configuration |
| `dspy.LM` | Model abstraction | Provider flexibility |

---

## Example Usage

```python
import dspy
from rlm import RecursiveLanguageModel, RLMConfig

# Configure
config = RLMConfig(
    root_model="openai/gpt-4o",
    worker_model="openai/gpt-4o-mini",
)
config.configure_dspy()

# Create RLM
rlm = RecursiveLanguageModel(config)

# Run a task
long_document = open("huge_document.txt").read()  # Millions of tokens!
result = rlm(
    task="Find all dates mentioned in the document and extract any events associated with them.",
    input_context=long_document
)

print(result.answer)
print(f"Token usage: {result.usage}")
```

---

## Advantages of DSPy Approach

1. **Less Prompt Engineering**: DSPy handles prompt construction automatically
2. **Optimization Ready**: Can use DSPy optimizers like `MIPROv2` to improve performance
3. **Cleaner Code**: Signatures and modules vs. string manipulation
4. **Built-in Tracking**: `track_usage=True` gives automatic token counting
5. **Multi-Provider**: Switch between OpenAI, Anthropic, Ollama, etc. with one line change
6. **Async Support**: Built-in async operations with `acall()` and `aforward()`

---

## Open Questions

1. **Custom ReAct vs. Built-in ReAct**: Should we extend `dspy.ReAct` or build custom loop?
   - *Decision*: Custom loop gives more control over LM() recursion

2. **Sub-LM Signature**: Should workers use ChainOfThought or just Predict?
   - *Decision*: Start with Predict, add CoT if quality suffers

3. **DSPy Optimization**: Can we use BootstrapFewShot or MIPROv2 to optimize?
   - *Experiment*: After MVP, try optimizing with training examples

4. **PythonInterpreter vs. Custom REPL**: Use DSPy's tool or build our own?
   - *Decision*: Custom REPL with INPUT variable and LM() function

---

## Next Steps

1. **Initialize project** with `uv init` and add `dspy` dependency
2. **Create config module** with dual LM setup
3. **Implement REPLState** with safety validation
4. **Build RecursiveLanguageModel** module with basic loop
5. **Add tools** one by one with unit tests
6. **Create example** - needle in haystack test
7. **Benchmark** and iterate on prompts/models
