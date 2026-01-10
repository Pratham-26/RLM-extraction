"""RLM Implementation - Recursive Language Model with Schema Extraction.

This package implements Recursive Language Models (RLMs) - a paradigm that enables
LLMs to process arbitrarily long prompts by treating the prompt as an external
environment that the LLM can programmatically interact with via Python code execution.
"""

from rlm.config import RLMConfig
from rlm.extract import RLMExtractor, ExtractionResult
from rlm.repl import REPLState

__version__ = "0.1.0"

__all__ = [
    "RLMConfig",
    "RLMExtractor",
    "ExtractionResult",
    "REPLState",
]
