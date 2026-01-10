"""RLM Configuration - Dual LM setup with modality routing."""

from dataclasses import dataclass, field
from typing import Literal

import dspy
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RLMConfig:
    """Configuration for RLM Schema Extraction.

    Supports dual-model architecture with routing based on input modality:
    - Root LM (orchestrator) - always text-based, does planning and coordination
    - Worker LM (extraction) - switches between text and vision models
    """

    # Model configuration
    root_model: str
    """Model for orchestration (always text-based)."""

    worker_text_model: str
    """Model for text document extraction."""

    worker_vision_model: str
    """Model for image document extraction (must be vision-capable)."""

    # Chunking
    chunk_size: int = 2000
    """Characters per text chunk (splits at nearest space)."""

    # Worker behavior
    summary_level: Literal["minimal", "standard", "verbose"] = "standard"
    """Detail level of chunk gists."""

    # Parallelism control
    parallel_first_pass: bool = True
    """Use parallel processing for initial extraction pass."""

    parallel_retry: bool = False
    """Use parallel processing for re-extraction retries."""

    max_parallel_workers: int = 5
    """Maximum concurrent API calls."""

    # Execution limits
    max_turns: int = 20
    """Maximum RLM orchestration turns."""

    code_execution_timeout: int = 30
    """Seconds before code execution timeout."""

    # API configuration
    api_key: str | None = None
    """API key (uses env var if not provided)."""

    # Internal state (filled by configure_dspy)
    _root_lm: dspy.LM = field(init=False, repr=False)
    _worker_text_lm: dspy.LM = field(init=False, repr=False)
    _worker_vision_lm: dspy.LM = field(init=False, repr=False)

    def configure_dspy(self) -> None:
        """Configure DSPy with the root LM as default."""
        api_key = self.api_key or _get_api_key_for_model(self.root_model)

        # Configure root LM (orchestrator)
        self._root_lm = dspy.LM(
            self.root_model,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.0,
        )

        # Configure worker LMs
        text_key = self.api_key or _get_api_key_for_model(self.worker_text_model)
        self._worker_text_lm = dspy.LM(
            self.worker_text_model,
            api_key=text_key,
            max_tokens=2048,
            temperature=0.0,
        )

        vision_key = self.api_key or _get_api_key_for_model(self.worker_vision_model)
        self._worker_vision_lm = dspy.LM(
            self.worker_vision_model,
            api_key=vision_key,
            max_tokens=2048,
            temperature=0.0,
        )

        # Set root LM as DSPy default
        dspy.configure(lm=self._root_lm, track_usage=True)

    def get_root_lm(self) -> dspy.LM:
        """Get the root LM instance."""
        if not hasattr(self, "_root_lm"):
            self.configure_dspy()
        return self._root_lm

    def get_worker_lm(self, modality: Literal["text", "vision"]) -> dspy.LM:
        """Get the appropriate worker LM for the input modality."""
        if not hasattr(self, "_worker_text_lm"):
            self.configure_dspy()

        if modality == "text":
            return self._worker_text_lm
        return self._worker_vision_lm


def _get_api_key_for_model(model: str) -> str:
    """Get the appropriate API key for a given model."""
    model_lower = model.lower()

    if "openai" in model_lower or model_lower.startswith("gpt"):
        import os

        return os.getenv("OPENAI_API_KEY", "")
    elif "anthropic" in model_lower or "claude" in model_lower:
        import os

        return os.getenv("ANTHROPIC_API_KEY", "")
    elif "openrouter" in model_lower:
        import os

        return os.getenv("OPENROUTER_API_KEY", "")

    # Default to OPENAI_API_KEY
    import os

    return os.getenv("OPENAI_API_KEY", "")


# Preset configurations

def openai_config() -> RLMConfig:
    """Pre-configured RLM for OpenAI models."""
    return RLMConfig(
        root_model="openai/gpt-4o",
        worker_text_model="openai/gpt-4o-mini",
        worker_vision_model="openai/gpt-4o",
    )


def anthropic_config() -> RLMConfig:
    """Pre-configured RLM for Anthropic models."""
    return RLMConfig(
        root_model="anthropic/claude-sonnet-4",
        worker_text_model="anthropic/claude-haiku-4",
        worker_vision_model="anthropic/claude-sonnet-4",
    )


def cost_optimized_config() -> RLMConfig:
    """Pre-configured RLM for cost optimization."""
    return RLMConfig(
        root_model="anthropic/claude-haiku-4",
        worker_text_model="anthropic/claude-haiku-4",
        worker_vision_model="openai/gpt-4o-mini",
    )


def quality_config() -> RLMConfig:
    """Pre-configured RLM for maximum quality."""
    return RLMConfig(
        root_model="anthropic/claude-sonnet-4",
        worker_text_model="openai/gpt-4o",
        worker_vision_model="anthropic/claude-sonnet-4",
    )
