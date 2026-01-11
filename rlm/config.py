"""RLM Configuration - Dual LM setup with modality routing."""

from dataclasses import dataclass, field
from typing import Literal

import dspy
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PDFConfig:
    """Configuration for PDF to image conversion."""

    # DPI for rendering (higher = better quality, larger images)
    dpi: int = 200

    # Page range: None = all pages, (start, end) for range, [n1, n2, ...] for specific pages
    page_range: tuple[int, int] | list[int] | None = None

    # Only process the first page
    first_page_only: bool = False

    # Image format for output (png, jpeg, etc.)
    fmt: str = "png"

    # Thread count for parallel conversion
    thread_count: int = 1


@dataclass
class RLMConfig:
    """Configuration for RLM Schema Extraction.

    Supports dual-model architecture with routing based on input modality:
    - Root LM (orchestrator) - always text-based, does planning and coordination
    - Worker LM (extraction) - switches between text and vision models
    """

    # Model configuration
    # Model for orchestration (always text-based)
    root_model: str

    # Model for text document extraction
    worker_text_model: str

    # Model for image document extraction (must be vision-capable)
    worker_vision_model: str

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
    max_turns: int = 20

    # Seconds before code execution timeout
    code_execution_timeout: int = 30

    # User context limits
    # Maximum characters allowed in user_context parameter
    max_user_context_chars: int = 10_000  # ~2,500 tokens

    # Retry limits
    # Maximum re-extraction attempts per chunk before giving up
    max_retries: int = 3

    # PDF processing mode
    # Controls how PDFs are processed: 'text', 'image', or 'auto'
    # - 'text': Extract text directly from PDF and process as text chunks
    # - 'image': Render PDF pages as images for vision models
    # - 'auto': Choose automatically based on file type (text/markdown → text, others → image)
    pdf_mode: Literal["text", "image", "auto"] = "auto"

    # PDF to image conversion settings
    pdf_config: PDFConfig = field(default_factory=PDFConfig)

    # Internal state (filled by configure_dspy)
    _root_lm: dspy.LM = field(init=False, repr=False)
    _worker_text_lm: dspy.LM = field(init=False, repr=False)
    _worker_vision_lm: dspy.LM = field(init=False, repr=False)

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
            max_tokens=4096,
            temperature=0.0,
        )

        # Configure worker LMs
        self._worker_text_lm = dspy.LM(
            self.worker_text_model,
            max_tokens=2048,
            temperature=0.0,
        )

        self._worker_vision_lm = dspy.LM(
            self.worker_vision_model,
            max_tokens=2048,
            temperature=0.0,
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

    def get_worker_lm(self, modality: Literal["text", "vision"]) -> dspy.LM:
        """Get the appropriate worker LM for the input modality.

        Args:
            modality: Either "text" or "vision"

        Returns:
            Configured worker LM for the specified modality
        """
        if not hasattr(self, "_worker_text_lm"):
            self.configure_dspy()

        if modality == "text":
            return self._worker_text_lm
        return self._worker_vision_lm


# Preset configurations


def openai_config() -> RLMConfig:
    """Pre-configured RLM for OpenAI models.

    Returns:
        RLMConfig configured with GPT-4o models
    """
    return RLMConfig(
        root_model="openai/gpt-4o",
        worker_text_model="openai/gpt-4o-mini",
        worker_vision_model="openai/gpt-4o",
    )


def anthropic_config() -> RLMConfig:
    """Pre-configured RLM for Anthropic models.

    Returns:
        RLMConfig configured with Claude models
    """
    return RLMConfig(
        root_model="anthropic/claude-sonnet-4",
        worker_text_model="anthropic/claude-haiku-4",
        worker_vision_model="anthropic/claude-sonnet-4",
    )


def cost_optimized_config() -> RLMConfig:
    """Pre-configured RLM for cost optimization.

    Returns:
        RLMConfig configured for minimal cost
    """
    return RLMConfig(
        root_model="anthropic/claude-haiku-4",
        worker_text_model="anthropic/claude-haiku-4",
        worker_vision_model="openai/gpt-4o-mini",
    )


def quality_config() -> RLMConfig:
    """Pre-configured RLM for maximum quality.

    Returns:
        RLMConfig configured for highest extraction quality
    """
    return RLMConfig(
        root_model="anthropic/claude-sonnet-4",
        worker_text_model="openai/gpt-4o",
        worker_vision_model="anthropic/claude-sonnet-4",
    )
