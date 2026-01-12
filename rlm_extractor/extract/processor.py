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
from rlm_extractor.logger import CallLogger
from rlm_extractor.signatures import WorkerExtractionSignature

if TYPE_CHECKING:
    pass


# Retry configuration
RETRY_DELAY = 1.0  # Fixed delay between retries (seconds)


@dataclass
class ChunkProcessingResult:
    """Result from processing a single chunk."""

    success: bool
    chunk_idx: int
    gist: str | None = None
    extracted: dict | None = None
    missing_fields: list[str] | None = None
    error: str | None = None


class ChunkProcessor:
    """Process document chunks with worker LMs, including retry logic."""

    def __init__(
        self,
        worker_lm: dspy.LM,
        max_parallel_workers: int = 5,
        max_attempts: int = 2,
        condensed_guidance: str = "",
        logger: CallLogger | None = None,
    ):
        self.worker_lm = worker_lm
        self.max_parallel_workers = max_parallel_workers
        self.max_attempts = max_attempts
        self.condensed_guidance = condensed_guidance
        self.logger = logger

        # Create DSPy predictor for worker with LM context
        with dspy.context(lm=self.worker_lm):
            self._worker_predictor = dspy.Predict(WorkerExtractionSignature)

    def _log_request(
        self, chunk: Chunk, yaml_schema: str, targeted_prompt: str, attempt: int
    ) -> str | None:
        """Log LLM request if logger available."""
        if not self.logger:
            return None

        request_payload = {
            "yaml_schema": yaml_schema,
            "chunk_idx": chunk.idx,
            "chunk_content": str(chunk.content)[:500],
            "condensed_guidance": self.condensed_guidance,
            "targeted_prompt": targeted_prompt,
        }
        metadata = {
            "chunk_idx": chunk.idx,
            "attempt": attempt,
            "max_attempts": self.max_attempts,
        }
        return self.logger.log_request(
            lm_type="worker",
            model=str(self.worker_lm),
            signature="WorkerExtractionSignature",
            call_type="worker_extraction",
            request=request_payload,
            metadata=metadata,
        )

    def _log_response(self, call_id: str | None, result: dspy.Prediction) -> None:
        """Log LLM response if logger available."""
        if self.logger and call_id:
            response_payload = {
                "gist": getattr(result, "gist", ""),
                "entity_contexts": getattr(result, "entity_contexts", ""),
                "missing_fields": getattr(result, "missing_fields", ""),
            }
            self.logger.log_response(call_id=call_id, response=response_payload)

    def _log_error(self, call_id: str | None, chunk: Chunk, error: str, attempt: int) -> None:
        """Log LLM error if logger available."""
        if self.logger and call_id:
            self.logger.log_error(
                call_id=call_id,
                error=error,
                metadata={"chunk_idx": chunk.idx, "attempt": attempt},
            )

    def process_chunk(
        self,
        chunk: Chunk,
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> ChunkProcessingResult:
        """Process a single chunk with retry logic."""
        current_prompt = targeted_prompt

        for attempt in range(1, self.max_attempts + 1):
            call_id = self._log_request(chunk, yaml_schema, current_prompt, attempt)

            try:
                result = self._worker_predictor(
                    yaml_schema=yaml_schema,
                    chunk_content=str(chunk.content),
                    chunk_idx=str(chunk.idx),
                    condensed_guidance=self.condensed_guidance,
                    targeted_prompt=current_prompt,
                )

                self._log_response(call_id, result)
                return self._parse_worker_result(result, chunk.idx)

            except (TimeoutError, Exception) as e:
                error_msg = str(e)
                self._log_error(call_id, chunk, error_msg, attempt)

                if attempt >= self.max_attempts:
                    return ChunkProcessingResult(
                        success=False,
                        chunk_idx=chunk.idx,
                        error=f"Failed after {attempt} attempts: {error_msg}",
                    )

                # Simple fixed delay before retry
                time.sleep(RETRY_DELAY)

                # Update prompt for retry
                current_prompt = self._get_retry_prompt(error_msg)

        # Should not reach here
        return ChunkProcessingResult(success=False, chunk_idx=chunk.idx, error="Unknown error")

    def _get_retry_prompt(self, error_msg: str) -> str:
        """Generate prompt for retry attempt."""
        if "parse" in error_msg.lower():
            return (
                "Please fix your output format. Previous attempt had parsing errors. "
                "Return entity contexts in 'field_name: description' format, one per line."
            )
        return f"Please try again. Previous attempt failed: {error_msg[:100]}"

    def _parse_worker_result(
        self, result: dspy.Prediction, chunk_idx: int
    ) -> ChunkProcessingResult:
        """Parse worker LM result into ChunkProcessingResult."""
        try:
            gist = getattr(result, "gist", "")
            entity_contexts_str = getattr(result, "entity_contexts", "")
            missing_fields_str = getattr(result, "missing_fields", "")

            return ChunkProcessingResult(
                success=True,
                chunk_idx=chunk_idx,
                gist=gist or f"Chunk {chunk_idx} processed",
                extracted=self._parse_entity_contexts(entity_contexts_str),
                missing_fields=self._parse_missing_fields(missing_fields_str),
            )

        except Exception as e:
            return ChunkProcessingResult(
                success=False,
                chunk_idx=chunk_idx,
                error=f"Failed to parse worker result: {str(e)}",
            )

    def _parse_entity_contexts(self, contexts_str: str) -> dict:
        """Parse 'field_name: description' format into dict."""
        if not contexts_str or contexts_str.strip() in ("", "none", "null"):
            return {}

        contexts = {}
        for line in contexts_str.strip().split("\n"):
            line = line.strip()
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
        """Parse missing fields string into list."""
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
                return [str(f).strip() for f in result if f][:max_missing_fields]
            return []
        except (ValueError, SyntaxError):
            if "," in cleaned:
                return [f.strip() for f in cleaned.split(",") if f][:max_missing_fields]
            single = cleaned.strip()
            return [single] if single else []

    def _error_result(self, chunk_idx: int, error: str) -> ChunkProcessingResult:
        """Create an error result."""
        return ChunkProcessingResult(success=False, chunk_idx=chunk_idx, error=error)

    def process_chunks_parallel(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompt: str = "",
    ) -> list[ChunkProcessingResult]:
        """Process multiple chunks in parallel."""
        results: list[ChunkProcessingResult] = []

        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, yaml_schema, targeted_prompt): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk, timeout=300):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except TimeoutError:
                    results.append(self._error_result(chunk.idx, "Timeout"))
                except Exception as e:
                    results.append(self._error_result(chunk.idx, str(e)))

        results.sort(key=lambda r: r.chunk_idx)
        return results

    def process_chunks_sequential(
        self,
        chunks: list[Chunk],
        yaml_schema: str,
        targeted_prompts: dict[int, str] | None = None,
    ) -> list[ChunkProcessingResult]:
        """Process chunks sequentially with optional targeted prompts."""
        results = []

        for chunk in chunks:
            prompt = ""
            if targeted_prompts and chunk.idx in targeted_prompts:
                prompt = targeted_prompts[chunk.idx]

            result = self.process_chunk(chunk, yaml_schema, prompt)
            results.append(result)

        return results
