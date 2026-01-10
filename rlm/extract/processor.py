"""Chunk Processor - Worker extraction with retry logic.

Handles parallel processing of document chunks by worker LMs,
including retry logic, error handling, and result aggregation.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

import dspy

from rlm.extract.chunker import Chunk


@dataclass
class ExtractionResult:
    """Result from processing a single chunk."""

    success: bool
    """Whether extraction succeeded."""

    chunk_idx: int
    """Index of the processed chunk."""

    gist: str | None = None
    """Summary of chunk content."""

    extracted: dict | None = None
    """Extracted data from this chunk."""

    confidence: str = "medium"
    """Confidence level: high/medium/low."""

    missing_fields: list[str] | None = None
    """Schema fields not found in this chunk."""

    error: str | None = None
    """Error message if extraction failed."""

    attempts: int = 1
    """Number of attempts made."""


class ChunkProcessor:
    """Process document chunks with worker LMs, including retry logic."""

    def __init__(
        self,
        worker_lm: dspy.LM,
        max_parallel_workers: int = 5,
        max_attempts: int = 2,
    ):
        """Initialize processor.

        Args:
            worker_lm: Worker LM for extraction
            max_parallel_workers: Maximum concurrent extractions
            max_attempts: Maximum retry attempts (default 2 = initial + 1 retry)
        """
        self.worker_lm = worker_lm
        self.max_parallel_workers = max_parallel_workers
        self.max_attempts = max_attempts

        # Create DSPy predictor for worker
        self._worker_predictor = None

    def _get_predictor(self) -> dspy.Predict:
        """Get or create worker predictor."""
        if self._worker_predictor is None:
            from rlm.signatures import WorkerExtractionSignature

            self._worker_predictor = dspy.Predict(WorkerExtractionSignature)
        return self._worker_predictor

    def process_chunk(
        self,
        chunk: Chunk,
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> ExtractionResult:
        """Process a single chunk with retry logic.

        Args:
            chunk: Chunk to process
            yaml_schema: YAML schema for extraction
            targeted_prompt: Optional specific instruction for re-extraction

        Returns:
            ExtractionResult with success status and data/error
        """
        predictor = self._get_predictor()

        for attempt in range(1, self.max_attempts + 1):
            try:
                # Call worker LM with current DSPy context
                with dspy.context(lm=self.worker_lm):
                    result = predictor(
                        yaml_schema=yaml_schema,
                        chunk_content=chunk.content,
                        chunk_idx=str(chunk.idx),
                        targeted_prompt=targeted_prompt,
                    )

                # Parse the result
                return self._parse_worker_result(result, chunk.idx, attempt)

            except TimeoutError as e:
                if attempt >= self.max_attempts:
                    return ExtractionResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Timeout after {attempt} attempts: {str(e)}",
                        attempts=attempt,
                    )
                # Retry for timeout without modifying prompt

            except Exception as e:
                error_msg = str(e)

                if attempt >= self.max_attempts:
                    return ExtractionResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Failed after {attempt} attempts: {error_msg}",
                        attempts=attempt,
                    )

                # Modify prompt for retry
                if "yaml" in error_msg.lower() or "parse" in error_msg.lower():
                    targeted_prompt = (
                        f"Please fix your YAML output. Previous attempt had parsing errors. "
                        f"Return only valid YAML with proper indentation."
                    )
                else:
                    targeted_prompt = f"Please try again. Previous attempt failed: {error_msg[:100]}"

        # Should not reach here
        return ExtractionResult(
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
    ) -> ExtractionResult:
        """Parse worker LM result into ExtractionResult."""
        try:
            # Extract fields from DSPy prediction
            gist = getattr(result, "gist", "")
            extracted_str = getattr(result, "extracted", "{}")
            confidence = getattr(result, "confidence", "medium").lower()
            missing_fields_str = getattr(result, "missing_fields", "")

            # Parse extracted YAML
            extracted = self._parse_extracted_yaml(extracted_str)

            # Parse missing fields
            missing_fields = self._parse_missing_fields(missing_fields_str)

            # Normalize confidence
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            return ExtractionResult(
                success=True,
                chunk_idx=chunk_idx,
                gist=gist or f"Chunk {chunk_idx} processed",
                extracted=extracted or {},
                confidence=confidence,
                missing_fields=missing_fields or [],
                attempts=attempts,
            )

        except Exception as e:
            return ExtractionResult(
                success=False,
                chunk_idx=chunk_idx,
                error=f"Failed to parse worker result: {str(e)}",
                attempts=attempts,
            )

    def _parse_extracted_yaml(self, yaml_str: str) -> dict:
        """Parse extracted YAML string into dict."""
        if not yaml_str or yaml_str.strip() == "{}":
            return {}

        try:
            import yaml

            return yaml.safe_load(yaml_str) or {}
        except Exception:
            # Try JSON fallback
            try:
                import json

                return json.loads(yaml_str)
            except Exception:
                return {}

    def _parse_missing_fields(self, fields_str: str) -> list[str]:
        """Parse missing fields string into list."""
        if not fields_str or fields_str.strip() == "[]" or fields_str.strip() == "":
            return []

        try:
            import ast

            result = ast.literal_eval(fields_str)
            if isinstance(result, list):
                return [str(f) for f in result]
            return []
        except Exception:
            # Try comma-separated
            if "," in fields_str:
                return [f.strip() for f in fields_str.split(",")]
            return [fields_str.strip()]

    def process_chunks_parallel(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> list[ExtractionResult]:
        """Process multiple chunks in parallel.

        Args:
            chunks: List of chunks to process
            yaml_schema: YAML schema for extraction
            targeted_prompt: Optional specific instruction (same for all chunks)

        Returns:
            List of ExtractionResult in same order as input chunks
        """
        results = [None] * len(chunks)

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
                    results[chunk.idx] = result
                except Exception as e:
                    results[chunk.idx] = ExtractionResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Unexpected error: {str(e)}",
                    )

        return results

    def process_chunks_sequential(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompts: dict[int, str] | None = None,
    ) -> list[ExtractionResult]:
        """Process chunks sequentially with optional targeted prompts.

        Args:
            chunks: List of chunks to process
            yaml_schema: YAML schema for extraction
            targeted_prompts: Optional dict mapping chunk_idx to specific prompt

        Returns:
            List of ExtractionResult
        """
        results = []

        for chunk in chunks:
            prompt = ""
            if targeted_prompts and chunk.idx in targeted_prompts:
                prompt = targeted_prompts[chunk.idx]

            result = self.process_chunk(chunk, yaml_schema, prompt)
            results.append(result)

        return results
