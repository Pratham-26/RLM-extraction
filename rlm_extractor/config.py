"""RLM Configuration - Dual LM setup for text-based extraction."""

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
    # Model for orchestration (always text-based)
    root_model: str

    # Model for text document extraction
    worker_text_model: str

    # Chunking
    # Characters per text chunk (splits at nearest space)
    chunk_size: int = 2000

    # Worker behavior
    # Detail level of chunk gists
    summary_level: Literal["minimal", "standard", "verbose"] = "standard"

    # Parallelism control
    # Use parallel processing for initial extraction pass
    parallel_first_pass: bool = True

    # Use parallel processing for re-extraction retries
    parallel_retry: bool = False

    # Maximum concurrent API calls
    max_parallel_workers: int = 5

    # Execution limits
    # Maximum RLM orchestration turns
    max_turns: int = 5

    # Seconds before code execution timeout
    code_execution_timeout: int = 30

    # User context limits
    # Maximum characters allowed in user_context parameter
    max_user_context_chars: int = 10_000  # ~2,500 tokens

    # Retry limits
    # Maximum re-extraction attempts per chunk before giving up
    max_retries: int = 3

    # Internal state (filled by configure_dspy)
    _root_lm: dspy.LM = field(init=False, repr=False)
    _worker_text_lm: dspy.LM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_summary_levels = {"minimal", "standard", "verbose"}
        if self.summary_level not in valid_summary_levels:
            raise ValueError(
                f"Invalid summary_level: {self.summary_level!r}. "
                f"Must be one of: {', '.join(sorted(valid_summary_levels))}"
            )

    def configure_dspy(self) -> None:
        """Configure DSPy with the root LM as default.

        DSPy uses litellm internally which will automatically read API keys
        from environment variables. Ensure your API keys are set before running.
        """
        # Configure root LM (orchestrator)
        self._root_lm = dspy.LM(
            self.root_model,
            max_tokens=8192,
            temperature=0.3,
        )

        # Configure worker LM
        self._worker_text_lm = dspy.LM(
            self.worker_text_model,
            max_tokens=4096,
            temperature=0.3,
        )

        # Set root LM as DSPy default
        dspy.configure(lm=self._root_lm, track_usage=True)

    def get_root_lm(self) -> dspy.LM:
        """Get the root LM instance.

        Returns:
            Configured root LM for orchestration
        """
        if not hasattr(self, "_root_lm"):
            self.configure_dspy()
        return self._root_lm

    def get_worker_lm(self) -> dspy.LM:
        """Get the worker LM instance.

        Returns:
            Configured worker LM for extraction
        """
        if not hasattr(self, "_worker_text_lm"):
            self.configure_dspy()
        return self._worker_text_lm
