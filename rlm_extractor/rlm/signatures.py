"""DSPy Signatures for RLM (Recursive Language Model).

Implements the paper's RLM paradigm with:
- Python REPL environment with INPUT variable
- LM() function for recursive self-calls
- FINAL() and FINAL_VAR() for returning answers
"""

import dspy


RLM_SYSTEM_PROMPT = """You are an assistant solving a task using a Python REPL environment.

## The Environment

The input context is available as the variable `INPUT` (a string).

## Available Actions

1. **Execute Python code**: Write code in ```python blocks to interact with INPUT
2. **Make sub-LM calls**: Use `LM(prompt)` to recursively invoke yourself on sub-tasks
3. **Return final answer**: Use `FINAL(answer)` or `FINAL_VAR(variable_name)`

## Guidelines

- **Be strategic**: Start by understanding what you need to accomplish
- **Use code effectively**: Python is your tool for analyzing, filtering, and processing INPUT
- **Chunk intelligently**: For long inputs, use LM() to process chunks in parallel
- **Verify when needed**: Use sub-calls to double-check your work
- **Manage context**: Direct string manipulation (regex, slicing) is cheaper than LM calls

## Python Capabilities

You have access to:
- The `INPUT` variable containing your document/context
- Standard string operations: slicing, `len()`, `in`, `.find()`, `.split()`, etc.
- The `re` module for regex pattern matching
- The `json` module for parsing JSON
- Common data structures: lists, dicts, sets, tuples
- `LM(prompt)` function to make recursive calls

## Sub-LM Usage

The `LM()` function creates a new RLM instance that:
- Receives your prompt as its INPUT variable
- Has the same capabilities as you (code execution, further LM calls)
- Returns its final answer as a string

Use sub-LMs for:
- Processing chunks of long input in parallel
- Having sub-tasks solved independently
- Verifying or cross-checking results

Example:
```python
# Process chunks in parallel
chunk_size = 5000
chunks = [INPUT[i:i+chunk_size] for i in range(0, len(INPUT), chunk_size)]
results = [LM(f"Extract all names from:\\n{chunk}") for chunk in chunks]
# Aggregate
all_names = []
for r in results:
    all_names.extend(r.split(","))
FINAL(",".join(set(all_names)))
```

## Output Formats

Use `FINAL(answer)` to return a direct answer:
```python
FINAL("The answer is 42")
```

Use `FINAL_VAR(variable_name)` to return the contents of a REPL variable:
```python
my_result = "complex computed value"
FINAL_VAR("my_result")
```

## Important Notes

- Variables persist across code executions within your session
- Use print() to see intermediate values (helpful for debugging)
- You can take multiple turns - don't try to do everything in one code block
- The system will keep executing your code until you call FINAL() or FINAL_VAR()
- Maximum recursion depth for LM() calls is limited - use them strategically

## Example Workflow

```python
# First, understand the input
print(f"Input length: {len(INPUT)}")
print(f"First 500 chars: {INPUT[:500]}")

# Look for specific patterns
import re
matches = re.findall(r'pattern', INPUT)
print(f"Found {len(matches)} matches")

# If complex processing needed, use sub-LMs
if len(INPUT) > 10000:
    answer = LM(f"Process this text:\\n{INPUT[:5000]}")
    FINAL(answer)
else:
    # Process directly with code
    result = "computed from INPUT"
    FINAL(result)
```
"""


class RLMSignature(dspy.Signature):
    """Main RLM interaction signature.

    The Root LM receives a task and orchestrates the solution through:
    1. Code execution on the INPUT variable
    2. Recursive LM() calls for sub-tasks
    3. FINAL() or FINAL_VAR() to return the answer

    This is NOT a traditional extraction signature - it's a general-purpose
    problem-solving interface using the RLM paradigm.
    """

    task = dspy.InputField(
        desc="The task or question to solve. "
        "The INPUT variable contains the document/context to work with."
    )

    conversation_history = dspy.InputField(
        desc="Previous turns in this conversation (code executed, results obtained). "
        "Empty on first turn.",
        default="",
    )

    response = dspy.OutputField(
        desc="Your response: either code in ```python blocks to execute, "
        "or FINAL(answer) / FINAL_VAR(variable) to return your final answer."
    )


class RLMToolUseSignature(dspy.Signature):
    """RLM signature for tool-augmented tasks.

    When the RLM needs access to additional tools beyond the basic REPL,
    this signature provides extended capabilities while maintaining the
    RLM paradigm.
    """

    task = dspy.InputField(
        desc="The task to solve using available tools and code execution"
    )

    available_tools = dspy.InputField(
        desc="Description of available tools beyond basic REPL (e.g., peek_context, "
        "web_search, file_read)",
        default="",
    )

    conversation_history = dspy.InputField(
        desc="Previous conversation turns",
        default="",
    )

    thought = dspy.OutputField(
        desc="Your reasoning about what to do next",
        default="",
    )

    action = dspy.OutputField(
        desc="Next action: 'code', 'tool', or 'final'",
    )

    code = dspy.OutputField(
        desc="Python code to execute if action='code'",
        default="",
    )

    tool_call = dspy.OutputField(
        desc="Tool invocation if action='tool'",
        default="",
    )

    final_answer = dspy.OutputField(
        desc="Final answer if action='final'",
        default="",
    )


# Backward compatibility signatures for extraction tasks
# These bridge the old extraction paradigm with the new RLM paradigm

class ExtractionTaskSignature(dspy.Signature):
    """Schema-based extraction using RLM paradigm.

    This signature adapts the RLM paradigm for structured extraction tasks.
    The Root LM uses code execution and LM() calls to orchestrate extraction.
    """

    task = dspy.InputField(
        desc="Extraction task description"
    )

    json_schema = dspy.InputField(
        desc="JSON Schema defining the structure of data to extract"
    )

    conversation_history = dspy.InputField(
        desc="Previous conversation turns",
        default="",
    )

    response = dspy.OutputField(
        desc="Your response: code to execute, or FINAL(json) with extracted data"
    )


class WorkerExtractionSignature(dspy.Signature):
    """Worker LM signature for processing document chunks.

    This is used by LM() sub-calls to process individual chunks.
    The worker extracts structured information from its assigned chunk.
    """

    task = dspy.InputField(
        desc="The extraction task"
    )

    json_schema = dspy.InputField(
        desc="JSON Schema for extraction"
    )

    chunk_content = dspy.InputField(
        desc="The document chunk to process"
    )

    chunk_index = dspy.InputField(
        desc="Index of this chunk (for reference)"
    )

    guidance = dspy.InputField(
        desc="Additional guidance from root LM",
        default="",
    )

    extracted_data = dspy.OutputField(
        desc="Extracted data in JSON format matching the schema"
    )

    confidence = dspy.OutputField(
        desc="Your confidence in this extraction: high, medium, or low",
        default="medium",
    )

    missing_fields = dspy.OutputField(
        desc="List of required fields that could not be found in this chunk",
        default="",
    )
