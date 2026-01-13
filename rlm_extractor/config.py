"""RLM Configuration - Dual LM setup for text-based extraction."""

import warnings

# Suppress Pydantic serialization warnings from litellm (must be before dspy import)
# litellm's internal Message schema (10 fields) doesn't match OpenAI's ChatCompletionMessage (7 fields)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")

from dataclasses import dataclass, field
from typing import Literal

import dspy
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RLMConfig:
    """Configuration for RLM Schema Extraction.

    Supports dual-model architecture:
    - Root LM (orchestrator) - text-based, does planning and coordination
    - Worker LM (extraction) - text-based extraction worker
    """

    # Model configuration
    root_model: str
    worker_text_model: str

    # LM parameters
    root_max_tokens: int = 65536
    worker_max_tokens: int = 8192
    temperature: float = 0.3

    # Chunking
    chunk_size: int = 2000

    # Worker behavior
    summary_level: Literal["minimal", "standard", "verbose"] = "standard"

    # Parallelism control
    parallel_first_pass: bool = True
    parallel_retry: bool = True  # Enable parallel retry instead of sequential one-at-a-time
    max_parallel_workers: int = 5

    # Execution limits
    max_turns: int = 5
    code_execution_timeout: int = 30
    chunk_timeout: int = 300  # Timeout per chunk in seconds (default: 5 minutes)

    # User context limits
    max_user_context_chars: int = 10_000

    # Retry limits
    max_retries: int = 3

    # Context efficiency
    compact_schema: bool = False  # Use compact YAML schema to save tokens

    # Root LM context compaction (prevents linear growth with chunk count)
    max_entity_contexts_per_field: int = 50  # Max contexts to send to Root LM per field
    max_chunk_summaries_for_root: int = 20  # Max chunk summaries to include
    enable_context_compaction: bool = True  # Use intelligent compaction vs truncation

    # Internal state
    _root_lm: dspy.LM = field(init=False, repr=False)
    _worker_text_lm: dspy.LM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration values and initialize LMs."""
        valid_summary_levels = {"minimal", "standard", "verbose"}
        if self.summary_level not in valid_summary_levels:
            raise ValueError(
                f"Invalid summary_level: {self.summary_level!r}. "
                f"Must be one of: {', '.join(sorted(valid_summary_levels))}"
            )
        self._configure_lms()

    def _create_lm(self, model_name: str, max_tokens: int) -> dspy.LM:
        """Create a DSPy LM instance."""
        return dspy.LM(model_name, max_tokens=max_tokens, temperature=self.temperature, cache=False)

    def _configure_lms(self) -> None:
        """Configure both LMs and set root as DSPy default."""
        self._root_lm = self._create_lm(self.root_model, self.root_max_tokens)
        self._worker_text_lm = self._create_lm(self.worker_text_model, self.worker_max_tokens)
        dspy.configure(lm=self._root_lm, track_usage=True)

    @property
    def root_lm(self) -> dspy.LM:
        """Get the root LM instance."""
        return self._root_lm

    @property
    def worker_lm(self) -> dspy.LM:
        """Get the worker LM instance."""
        return self._worker_text_lm
