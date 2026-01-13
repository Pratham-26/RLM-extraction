"""RLM Extractor - Recursive Language Model with Schema Extraction.

This package implements Recursive Language Models (RLMs) - a paradigm that enables
LLMs to process arbitrarily long prompts by treating the prompt as an external
environment that the LLM can programmatically interact with via Python code execution.

## Architecture

The RLM paradigm provides:
1. Python REPL environment with INPUT variable (the document)
2. LM() function for recursive self-calls
3. FINAL() / FINAL_VAR() for returning answers
4. Code execution with safety guards

## Usage

Basic extraction:
    >>> from rlm_extractor import extract
    >>> result = extract(
    ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
    ...     document="John Doe is here.",
    ...     root_model="openai/gpt-4o",
    ...     worker_text_model="openai/gpt-4o-mini",
    ... )
    >>> print(result.data)
    {'name': 'John Doe'}

Advanced RLM usage:
    >>> from rlm_extractor.rlm import RLMExecutor, RLMConfig
    >>> executor = RLMExecutor(lm=your_dspy_lm, config=RLMConfig(max_turns=10))
    >>> result = executor.run(
    ...     task="Find all names in the document",
    ...     input_context=document,
    ... )
    >>> print(result.answer)
"""

import warnings

# Suppress Pydantic serialization warnings from litellm
# These warnings occur due to schema mismatch between litellm's internal Message model
# (10 fields) and OpenAI's ChatCompletionMessage (7 fields). This is an upstream
# litellm issue that does not affect functionality.
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")

# Core RLM implementation (NEW - paper-compliant)
from rlm_extractor.rlm import (
    RLMExecutor,
    RLMExtractor as NewRLMExtractor,
    RLMConfig as NewRLMConfig,
    RLMREPL,
    RLMEnvironment,
    SubLMHandler,
    OutputParser,
    RLMResult,
    SafeREPL,
    SafetyError,
    RLM_SYSTEM_PROMPT,
)

# Legacy extraction (OLD - for backward compatibility)
from rlm_extractor.config import RLMConfig as LegacyRLMConfig
from rlm_extractor.extract import ExtractionResult, RLMExtractor as LegacyRLMExtractor
from rlm_extractor.logger import CallLogger
from rlm_extractor.repl import REPLState

__version__ = "0.2.0"


def extract(
    schema: dict,
    document: str | list,
    root_model: str,
    worker_text_model: str,
    task: str | None = None,
    user_context: str | None = None,
    use_rlm_paradigm: bool = True,
    **config_kwargs,
) -> ExtractionResult:
    """One-shot extraction with specified models.

    Simplest way to use rlm_extractor - no config setup required.
    Specify your models and let RLM handle the extraction.

    Args:
        schema: JSON Schema defining what to extract
        document: Text string or list of file paths
        root_model: Model for orchestration (e.g., "openai/gpt-4o")
        worker_text_model: Model for document extraction
        task: Optional custom task description
        user_context: Optional user-provided context and instructions
        use_rlm_paradigm: If True, use new RLM paradigm (code execution + recursion).
                         If False, use legacy extraction orchestrator.
        **config_kwargs: Override any config setting

    Returns:
        ExtractionResult with data, gists, failures, and usage

    Example:
        >>> from rlm_extractor import extract
        >>> result = extract(
        ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ...     document="John Doe is here.",
        ...     root_model="openai/gpt-4o",
        ...     worker_text_model="openai/gpt-4o-mini",
        ... )
        >>> print(result.data)
        {'name': 'John Doe'}
    """
    if use_rlm_paradigm:
        # Use new RLM paradigm
        import dspy
        import json

        config = NewRLMConfig(**{k: v for k, v in config_kwargs.items() if hasattr(NewRLMConfig, k) or k in NewRLMConfig.__dataclass_fields__})

        root_lm = dspy.LM(root_model, max_tokens=8192)
        worker_lm = dspy.LM(worker_text_model, max_tokens=4096)

        executor = RLMExecutor(lm=root_lm, config=config)

        # Default task
        if task is None:
            task = f"""Extract structured data from the INPUT document according to this JSON Schema:

{json.dumps(schema, indent=2)}

Return the extracted data as FINAL(json_string) where json_string contains only the JSON (no markdown wrapping)."""

        result = executor.run(task=task, input_context=document if isinstance(document, str) else str(document))

        # Convert to ExtractionResult format
        return ExtractionResult(
            data={"raw_answer": result.answer} if not result.success else _parse_json_answer(result.answer),
            chunk_gists=[],
            failures=[],
            turns=result.turns,
            token_usage={},
            log_file_path=None,
        )
    else:
        # Use legacy extraction
        config = LegacyRLMConfig(
            root_model=root_model,
            worker_text_model=worker_text_model,
        )
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        extractor = LegacyRLMExtractor(config)
        return extractor.extract(
            json_schema=schema,
            document=document,
            task=task,
            user_context=user_context,
        )


def _parse_json_answer(answer: str) -> dict:
    """Parse JSON from answer, handling various formats."""
    import json
    import re

    answer = answer.strip()

    # Remove FINAL() wrapper if present
    if answer.startswith("FINAL("):
        answer = answer[6:-1]

    # Remove markdown code blocks
    if "```json" in answer:
        match = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
        if match:
            answer = match.group(1)
    elif "```" in answer:
        match = re.search(r"```\s*(.*?)\s*```", answer, re.DOTALL)
        if match:
            answer = match.group(1)

    # Try parsing
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return {"raw_answer": answer}


# Re-exports for backward compatibility
RLMConfig = LegacyRLMConfig  # Default to legacy for backward compat
RLMExtractor = LegacyRLMExtractor  # Default to legacy for backward compat


__all__ = [
    # New RLM paradigm (paper-compliant)
    "RLMExecutor",
    "NewRLMExtractor",
    "NewRLMConfig",
    "RLMREPL",
    "RLMEnvironment",
    "SubLMHandler",
    "OutputParser",
    "RLMResult",
    "SafeREPL",
    "SafetyError",
    "RLM_SYSTEM_PROMPT",
    # Legacy (backward compatibility)
    "RLMConfig",
    "RLMExtractor",
    "ExtractionResult",
    "CallLogger",
    "REPLState",
    # Convenience
    "extract",
]
