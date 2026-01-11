"""RLM Extractor - Recursive Language Model with Schema Extraction.

This package implements Recursive Language Models (RLMs) - a paradigm that enables
LLMs to process arbitrarily long prompts by treating the prompt as an external
environment that the LLM can programmatically interact with via Python code execution.
"""

from rlm_extractor.config import RLMConfig
from rlm_extractor.extract import ExtractionResult, RLMExtractor
from rlm_extractor.repl import REPLState

__version__ = "0.1.0"


def extract(
    schema: dict,
    document: str | list,
    root_model: str,
    worker_text_model: str,
    worker_vision_model: str,
    task: str | None = None,
    user_context: str | None = None,
    **config_kwargs,
) -> ExtractionResult:
    """One-shot extraction with specified models.

    Simplest way to use rlm_extractor - no config setup required.
    Specify your models and let RLM handle the extraction.

    Args:
        schema: JSON Schema defining what to extract
        document: Text string, list of PIL Images, or list of file paths
        root_model: Model for orchestration (e.g., "openrouter/anthropic/claude-sonnet-4")
        worker_text_model: Model for text document extraction
        worker_vision_model: Model for image document extraction (must be vision-capable)
        task: Optional custom task description
        user_context: Optional user-provided context and instructions
        **config_kwargs: Override any RLMConfig setting (chunk_size, max_parallel_workers, etc.)

    Returns:
        ExtractionResult with data, gists, failures, and usage

    Example:
        >>> from rlm_extractor import extract
        >>> result = extract(
        ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ...     document="John Doe is here.",
        ...     root_model="openrouter/anthropic/claude-sonnet-4",
        ...     worker_text_model="openrouter/anthropic/claude-haiku-4",
        ...     worker_vision_model="openrouter/anthropic/claude-sonnet-4",
        ... )
        >>> print(result.data)
        {'name': 'John Doe'}
    """
    config = RLMConfig(
        root_model=root_model,
        worker_text_model=worker_text_model,
        worker_vision_model=worker_vision_model,
    )
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    extractor = RLMExtractor(config)
    return extractor.extract(
        json_schema=schema,
        document=document,
        task=task,
        user_context=user_context,
    )


__all__ = [
    "RLMConfig",
    "RLMExtractor",
    "ExtractionResult",
    "REPLState",
    "extract",
]
