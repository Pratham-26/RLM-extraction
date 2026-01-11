"""Chunk Processor - Worker extraction with retry logic.

Handles parallel processing of document chunks by worker LMs,
including retry logic, error handling, and result aggregation.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dspy

from rlm_extractor.extract.chunker import Chunk

if TYPE_CHECKING:
    pass


# Retry configuration constants
class RetryConfig:
    # Base delay for exponential backoff (seconds)
    BASE_DELAY: float = 1.0

    # Maximum delay between retries (seconds)
    MAX_DELAY: float = 10.0

    # Backoff multiplier for exponential increase
    BACKOFF_MULTIPLIER: float = 2.0


@dataclass
class ChunkProcessingResult:
    """Result from processing a single chunk."""

    # Whether extraction succeeded
    success: bool

    # Index of the processed chunk
    chunk_idx: int

    # Summary of chunk content
    gist: str | None = None

    # Extracted data from this chunk
    extracted: dict | None = None

    # Confidence level: high/medium/low
    confidence: str = "medium"

    # Schema fields not found in this chunk
    missing_fields: list[str] | None = None

    # Error message if extraction failed
    error: str | None = None

    # Number of attempts made
    attempts: int = 1


class ChunkProcessor:
    """Process document chunks with worker LMs, including retry logic."""

    def __init__(
        self,
        worker_lm: dspy.LM,
        max_parallel_workers: int = 5,
        max_attempts: int = 2,
        condensed_guidance: str = "",
    ):
        """Initialize processor.

        Args:
            worker_lm: Worker LM for extraction
            max_parallel_workers: Maximum concurrent extractions
            max_attempts: Maximum retry attempts (default 2 = initial + 1 retry)
            condensed_guidance: Condensed user guidance for workers
        """
        self.worker_lm = worker_lm
        self.max_parallel_workers = max_parallel_workers
        self.max_attempts = max_attempts
        self.condensed_guidance = condensed_guidance

        # Create DSPy predictor for worker with LM context
        with dspy.context(lm=self.worker_lm):
            from rlm_extractor.signatures import WorkerExtractionSignature

            self._worker_predictor = dspy.Predict(WorkerExtractionSignature)

    def _prepare_chunk_content(self, chunk: Chunk):
        """Prepare chunk content for DSPy processing.

        Wraps PIL images in dspy.Image for proper vision model handling.
        Text chunks are passed through unchanged.

        Args:
            chunk: Chunk with content (text string, PIL Image, or dspy.Image)

        Returns:
            Prepared content: text string or dspy.Image
        """
        if TYPE_CHECKING:
            from PIL import Image as PILImage
        else:
            try:
                from PIL import Image as PILImage
            except ImportError:
                PILImage = None

        if PILImage and isinstance(chunk.content, PILImage.Image):
            # Wrap PIL image in dspy.Image for vision processing
            # DSPy handles encoding and formatting internally
            return dspy.Image(chunk.content)
        elif isinstance(chunk.content, dspy.Image):
            # Already a dspy.Image, pass through
            return chunk.content
        else:
            # Text chunk, pass through as-is
            return chunk.content

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay for a given attempt.

        Args:
            attempt: The attempt number (1-indexed)

        Returns:
            Delay in seconds, capped at MAX_DELAY
        """
        # Calculate exponential backoff: base * (multiplier ^ (attempt - 1))
        delay = RetryConfig.BASE_DELAY * (RetryConfig.BACKOFF_MULTIPLIER ** (attempt - 1))
        return min(delay, RetryConfig.MAX_DELAY)

    def process_chunk(
        self,
        chunk: Chunk,
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> ChunkProcessingResult:
        """Process a single chunk with retry logic.

        Args:
            chunk: Chunk to process
            yaml_schema: YAML schema for extraction
            targeted_prompt: Optional specific instruction for re-extraction

        Returns:
            ChunkProcessingResult with success status and data/error
        """

        # Prepare chunk content for DSPy
        prepared_content = self._prepare_chunk_content(chunk)

        for attempt in range(1, self.max_attempts + 1):
            try:
                # Call worker LM with predictor (already has LM configured)
                result = self._worker_predictor(
                    yaml_schema=yaml_schema,
                    chunk_content=prepared_content,
                    chunk_idx=str(chunk.idx),
                    condensed_guidance=self.condensed_guidance,
                    targeted_prompt=targeted_prompt,
                )

                # Parse result
                return self._parse_worker_result(result, chunk.idx, attempt)

            except TimeoutError as e:
                if attempt >= self.max_attempts:
                    return ChunkProcessingResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Timeout after {attempt} attempts: {str(e)}",
                        attempts=attempt,
                    )
                # Apply exponential backoff before retry
                time.sleep(self._calculate_backoff(attempt))

            except Exception as e:
                error_msg = str(e)

                if attempt >= self.max_attempts:
                    return ChunkProcessingResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Failed after {attempt} attempts: {error_msg}",
                        attempts=attempt,
                    )

                # Apply exponential backoff before retry
                time.sleep(self._calculate_backoff(attempt))

                # Modify prompt for retry
                if "parse" in error_msg.lower():
                    targeted_prompt = (
                        "Please fix your output format. Previous attempt had parsing errors. "
                        "Return entity contexts in 'field_name: description' format, one per line."
                    )
                else:
                    targeted_prompt = (
                        f"Please try again. Previous attempt failed: {error_msg[:100]}"
                    )

        # Should not reach here
        return ChunkProcessingResult(
            success=False,
            chunk_idx=chunk.idx,
            error="Unknown error",
            attempts=self.max_attempts,
        )

    def _parse_worker_result(
        self,
        result: dspy.Prediction,
        chunk_idx: int,
        attempts: int,
    ) -> ChunkProcessingResult:
        """Parse worker LM result into ChunkProcessingResult."""
        try:
            # Extract fields from DSPy prediction
            gist = getattr(result, "gist", "")
            entity_contexts_str = getattr(result, "entity_contexts", "")
            confidence_val = getattr(result, "confidence", "medium")
            confidence = (
                confidence_val.lower() if isinstance(confidence_val, str) else confidence_val
            )
            missing_fields_str = getattr(result, "missing_fields", "")

            # Parse entity contexts
            entity_contexts = self._parse_entity_contexts(entity_contexts_str)

            # Parse missing fields
            missing_fields = self._parse_missing_fields(missing_fields_str)

            # Normalize confidence
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            return ChunkProcessingResult(
                success=True,
                chunk_idx=chunk_idx,
                gist=gist or f"Chunk {chunk_idx} processed",
                extracted=entity_contexts or {},
                confidence=confidence,
                missing_fields=missing_fields or [],
                attempts=attempts,
            )

        except Exception as e:
            return ChunkProcessingResult(
                success=False,
                chunk_idx=chunk_idx,
                error=f"Failed to parse worker result: {str(e)}",
                attempts=attempts,
            )

    def _parse_entity_contexts(self, contexts_str: str) -> dict:
        """Parse 'field_name: description' format into dict.

        Simple parsing - no schema validation needed. Worker just describes
        what it sees; Root LM handles schema mapping and validation.

        Args:
            contexts_str: Raw entity contexts from worker LM

        Returns:
            Dict mapping field_name -> context_description
        """
        if not contexts_str or contexts_str.strip() in ("", "none", "null"):
            return {}

        contexts = {}
        lines = contexts_str.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Skip empty lines and common LLM artifacts
            if not line or line.lower().startswith(
                ("entity contexts:", "entity_contexts:", "contexts:")
            ):
                continue
            if ":" not in line:
                continue

            # Split on first colon only
            field_name = line[: line.index(":")].strip()
            description = line[line.index(":") + 1 :].strip()

            if field_name:
                contexts[field_name] = description

        return contexts

    def _parse_missing_fields(self, fields_str: str) -> list[str]:
        """Parse missing fields string into list.

        Handles list format from LLM output (e.g., "['field1', 'field2']")
        and comma-separated as fallback. Input is sanitized to prevent
        injection attacks.
        """
        # Maximum number of fields to prevent abuse
        max_missing_fields = 100

        if not fields_str or fields_str.strip() in ("[]", "", "none", "null"):
            return []

        # Strip common LLM artifacts
        cleaned = fields_str.strip()
        for prefix in ("Missing fields:", "missing:", "fields:"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        try:
            import ast

            result = ast.literal_eval(cleaned)
            if isinstance(result, list):
                # Limit size and sanitize each field
                fields = [str(f).strip() for f in result if f]
                return fields[:max_missing_fields]
            return []
        except (ValueError, SyntaxError):
            # Try comma-separated as fallback
            if "," in cleaned:
                fields = [f.strip() for f in cleaned.split(",")]
                # Filter out empty strings and limit
                return [f for f in fields if f][:max_missing_fields]
            # Single field
            single = cleaned.strip()
            return [single] if single else []

    def process_chunks_parallel(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> list[ChunkProcessingResult]:
        """Process multiple chunks in parallel.

        Args:
            chunks: List of chunks to process
            yaml_schema: YAML schema for extraction
            targeted_prompt: Optional specific instruction (same for all chunks)

        Returns:
            List of ChunkProcessingResult in same order as input chunks
        """
        results: list[ChunkProcessingResult] = []

        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all jobs
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, yaml_schema, targeted_prompt): chunk
                for chunk in chunks
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        ChunkProcessingResult(
                            success=False,
                            chunk_idx=chunk.idx,
                            error=f"Unexpected error: {str(e)}",
                        )
                    )

        # Sort results by chunk index
        results.sort(key=lambda r: r.chunk_idx)
        return results

    def process_chunks_sequential(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompts: dict[int, str] | None = None,
    ) -> list[ChunkProcessingResult]:
        """Process chunks sequentially with optional targeted prompts.

        Args:
            chunks: List of chunks to process
            yaml_schema: YAML schema for extraction
            targeted_prompts: Optional dict mapping chunk_idx to specific prompt

        Returns:
            List of ChunkProcessingResult
        """
        results = []

        for chunk in chunks:
            prompt = ""
            if targeted_prompts and chunk.idx in targeted_prompts:
                prompt = targeted_prompts[chunk.idx]

            result = self.process_chunk(chunk, yaml_schema, prompt)
            results.append(result)

        return results
