"""RLMExtractor - Main orchestrator for schema-based extraction.

Coordinates the entire extraction process:
1. Converts JSON Schema to YAML
2. Chunks the document
3. Runs parallel first-pass extraction
4. Orchestrates sequential re-extraction as needed
5. Aggregates results and converts back to JSON
"""

import json
import os
import re
from dataclasses import dataclass, field

import dspy

from rlm_extractor.config import RLMConfig
from rlm_extractor.extract.chunker import (
    ALL_SUPPORTED_EXTENSIONS,
    Chunk,
    Chunker,
)
from rlm_extractor.extract.processor import ChunkProcessingResult as ChunkResult
from rlm_extractor.extract.processor import ChunkProcessor
from rlm_extractor.extract.schema import SchemaConverter
from rlm_extractor.logger import CallLogger
from rlm_extractor.repl import REPLState
from rlm_extractor.signatures import RootExtractionSignature

# Constants for user_context validation
MIN_USER_CONTEXT_CHARS = 10


@dataclass
class ExtractionResult:
    """Result of RLM schema extraction."""

    # Final extracted data matching the input JSON Schema
    data: dict

    # Summary of each chunk: [{idx, gist, confidence, fields_found}]
    chunk_gists: list[dict] = field(default_factory=list)

    # Failed chunks: [{chunk_idx, error, content_preview}]
    failures: list[dict] = field(default_factory=list)

    # Total number of RLM orchestration turns
    turns: int = 0

    # Token consumption: {root: int, worker: int, total: int}
    token_usage: dict = field(default_factory=dict)

    # Path to LLM call log file (JSON Lines format)
    log_file_path: str | None = None

    def is_complete(self) -> bool:
        """True if extraction has no failures."""
        return len(self.failures) == 0

    def get_failure_rate(self) -> float:
        """Percentage of chunks that failed (0.0 to 1.0)."""
        if not self.chunk_gists:
            return 0.0
        return len(self.failures) / len(self.chunk_gists)


class RLMExtractor(dspy.Module):
    """Schema-based extraction using RLM paradigm.

    Usage:
        config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
        )
        extractor = RLMExtractor(config)

        result = extractor.extract(
            json_schema={"type": "object", "properties": {...}},
            document="large document text...",
        )
    """

    def __init__(self, config: RLMConfig):
        """Initialize extractor with dual LM configuration.

        Args:
            config: RLMConfig with model settings
        """
        super().__init__()
        self.config = config

        # Configure DSPy
        if not hasattr(config, "_root_lm"):
            config.configure_dspy()

        # Initialize components
        self.schema_converter = SchemaConverter()
        self.chunker = Chunker(chunk_size=config.chunk_size)
        self.repl = REPLState(summary_level=config.summary_level)

        # Root LM predictor
        self.root_predictor = dspy.Predict(RootExtractionSignature)

    def extract(
        self,
        json_schema: dict,
        document: str | list[str],
        task: str | None = None,
        user_context: str | None = None,
    ) -> ExtractionResult:
        """Extract structured data from a document according to JSON Schema.

        Args:
            json_schema: JSON Schema defining what to extract
            document: Text string or list of file paths (.txt, .md, or .pdf)
            task: Optional custom task description
            user_context: Optional user-provided context and instructions.
                The Root LM will condense this into extraction guidance for workers.

        Returns:
            ExtractionResult with data, gists, failures, and usage

        Raises:
            ValueError: If inputs are invalid
            TypeError: If document types are incorrect
        """
        # Validate inputs
        self._validate_inputs(json_schema, document)

        # Get worker LM
        worker_lm = self.config.get_worker_lm()

        # Initialize logger
        logger = CallLogger()

        # Convert schema
        yaml_schema = self.schema_converter.json_to_yaml_chunks(json_schema)

        # Chunk the document
        chunks = self._chunk_document(document)

        # Initialize REPL state
        input_context = (
            document if isinstance(document, str)
            else [c.content for c in chunks]
        )
        self.repl.reset_for_task(
            input_context=input_context,
            yaml_schema=yaml_schema,
        )
        self.repl.set_total_chunks(len(chunks))

        # Store JSON schema for field tracking
        self.repl.set_json_schema(json_schema)

        # Condense user context if provided
        condensed_guidance = ""
        if user_context:
            self._validate_user_context(user_context)
            sanitized_context = self._sanitize_user_context(user_context)
            condensed_guidance = self._condense_user_context(sanitized_context, yaml_schema, logger)
            self.repl.set_condensed_guidance(condensed_guidance)

        # Create chunk processor
        processor = ChunkProcessor(
            worker_lm=worker_lm,
            max_parallel_workers=self.config.max_parallel_workers,
            condensed_guidance=condensed_guidance,
            logger=logger,
        )

        # Set default task
        if task is None:
            task = "Extract all fields from the document according to the schema."

        # Main orchestration loop
        trajectory = []

        for turn in range(self.config.max_turns):
            # First turn: run parallel extraction
            if turn == 0 and self.config.parallel_first_pass:
                results = processor.process_chunks_parallel(chunks, yaml_schema)
                # Process results
                self._process_worker_results(results, chunks)
                trajectory.append(self._create_trajectory_entry(results))

                # Calculate success rate
                success_count = sum(1 for r in results if r.success)
                success_rate = success_count / max(len(results), 1)

                # If high success rate, skip root LM and finalize immediately
                # This is the most common case and avoids expensive root orchestration
                if success_rate >= 0.8:
                    break

            else:
                # Only enter root LM decision loop if first pass had issues
                action_result = self._process_root_decision(
                    task, trajectory, yaml_schema, processor, chunks, logger
                )
                trajectory.append({"action": action_result, "turn": turn})

                if action_result == "finalize":
                    break

                # Force finalize after max turns to prevent infinite loops
                if turn >= self.config.max_turns - 1:
                    break

        # Convert results back to JSON
        final_json = self.schema_converter.yaml_to_json(
            json.dumps(self.repl.results_so_far),
            json_schema,
        )

        # Compile failures
        failures = self._compile_failures(chunks)

        # Get token usage
        usage = self._get_token_usage()

        # Close logger and get file path
        logger.close()
        log_path = str(logger.get_log_file_path())

        return ExtractionResult(
            data=final_json,
            chunk_gists=self.repl.chunk_summaries,
            failures=failures,
            turns=turn + 1,
            token_usage=usage,
            log_file_path=log_path,
        )

    def _validate_inputs(
        self,
        json_schema: dict,
        document: str | list[str],
    ) -> None:
        """Validate input parameters.

        Raises:
            ValueError: If inputs are invalid
            TypeError: If document types are incorrect
        """
        # Validate json_schema
        if json_schema is None:
            raise ValueError("json_schema cannot be None")
        if not isinstance(json_schema, dict):
            raise TypeError(f"json_schema must be a dict, got {type(json_schema).__name__}")
        if not json_schema:
            raise ValueError("json_schema cannot be empty")
        if "type" not in json_schema:
            raise ValueError("json_schema must have a 'type' field (e.g., 'type': 'object')")

        # Validate document
        if document is None:
            raise ValueError("document cannot be None")

        if isinstance(document, str):
            # File paths are allowed, so only check if it's empty content (not a file)
            if not document.strip():
                raise ValueError("document string cannot be empty")
            # If it looks like a file path but doesn't exist, that's an error
            if os.path.exists(document) or os.path.exists(os.path.abspath(document)):
                # File exists - validate extension
                abs_path = os.path.abspath(document)
                ext = os.path.splitext(abs_path)[1].lower()
                if ext not in ALL_SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported file type: {ext}. "
                        f"Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
                    )
        elif isinstance(document, list):
            if len(document) == 0:
                raise ValueError("document list cannot be empty")

            # Validate list element types
            first = document[0]
            if not isinstance(first, str):
                raise TypeError(
                    f"document list must contain str (file paths), "
                    f"got {type(first).__name__}"
                )

            # Check all elements are same type
            for i, item in enumerate(document):
                if type(item) is not type(first):
                    raise TypeError(
                        f"document list must contain consistent types; "
                        f"element 0 is {type(first).__name__} but element {i} is {type(item).__name__}"
                    )
        else:
            raise TypeError(
                f"document must be str or list[str] (file paths), "
                f"got {type(document).__name__}"
            )

    def _validate_user_context(self, user_context: str) -> None:
        """Validate user_context parameter.

        Args:
            user_context: User-provided context string

        Raises:
            TypeError: If user_context is not a string
            ValueError: If user_context is empty, too short, or too long
        """
        if not isinstance(user_context, str):
            raise TypeError(f"user_context must be a string, got {type(user_context).__name__}")

        if len(user_context) == 0:
            raise ValueError("user_context cannot be empty")

        if len(user_context) < MIN_USER_CONTEXT_CHARS:
            raise ValueError(
                f"user_context is too short ({len(user_context)} chars). "
                f"Minimum: {MIN_USER_CONTEXT_CHARS} chars. "
                f"If you don't need additional context, omit the parameter."
            )

        max_chars = self.config.max_user_context_chars
        if len(user_context) > max_chars:
            raise ValueError(
                f"user_context is too long ({len(user_context)} chars). "
                f"Maximum: {max_chars:,} chars. "
                f"You can increase this by setting max_user_context_chars in RLMConfig."
            )

    def _sanitize_user_context(self, user_context: str) -> str:
        """Sanitize user context to reduce injection risk.

        Args:
            user_context: Raw user-provided context

        Returns:
            Sanitized context string
        """
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", user_context)
        # Limit repeated newlines (max 2 consecutive)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _condense_user_context(
        self, user_context: str, yaml_schema: str, logger: CallLogger
    ) -> str:
        """Have Root LM condense user context into extraction guidance.

        Args:
            user_context: User-provided context and instructions
            yaml_schema: YAML schema for extraction
            logger: CallLogger instance

        Returns:
            Condensed guidance (2-4 sentences) for Worker LMs
        """
        from rlm_extractor.signatures import ContextCondensationSignature

        condenser = dspy.Predict(ContextCondensationSignature)

        # Log request
        root_lm = self.config.get_root_lm()
        call_id = logger.log_request(
            lm_type="root",
            model=str(root_lm),
            signature="ContextCondensationSignature",
            call_type="context_condensation",
            request={"user_context": user_context, "yaml_schema": yaml_schema},
        )

        with dspy.context(lm=root_lm):
            result = condenser(user_context=user_context, yaml_schema=yaml_schema)

        # Log response
        logger.log_response(
            call_id=call_id,
            response={"condensed_guidance": getattr(result, "condensed_guidance", "")},
        )

        return getattr(result, "condensed_guidance", "")

    def _is_file_path(self, document: str) -> bool:
        """Check if a string is a file path vs. document content.

        Args:
            document: String to check

        Returns:
            True if the string appears to be a valid file path
        """
        if not document or len(document) > 1024:
            # Reasonable path length limit
            return False

        # Check if file exists
        if os.path.exists(document):
            return True

        # Check if absolute path exists
        abs_path = os.path.abspath(document)
        if os.path.exists(abs_path):
            return True

        return False

    def _chunk_document(
        self,
        document: str | list[str],
    ) -> list:
        """Chunk the document for processing."""
        if isinstance(document, str):
            # Check if it's a file path
            if self._is_file_path(document):
                return self.chunker.chunk_file(document)
            return self.chunker.chunk_text(document)
        else:
            # List of file paths
            chunks = []
            next_idx = 0
            for path in document:
                file_chunks = self.chunker.chunk_file(path)
                # Adjust chunk indices to maintain sequential order
                for chunk in file_chunks:
                    chunk.idx = next_idx
                    next_idx += 1
                chunks.extend(file_chunks)
            return chunks

    def _process_worker_results(
        self,
        results: list[ChunkResult],
        chunks: list,
    ) -> None:
        """Process results from workers and update REPL state."""
        for result in results:
            if result.success and result.extracted:
                self.repl.update_chunk_result(
                    idx=result.chunk_idx,
                    gist=result.gist or "",
                    extracted=result.extracted,
                    confidence=result.confidence,
                    fields_found=list(result.extracted.keys()),
                )
            elif not result.success:
                # Mark as failed
                self.repl.mark_failed(result.chunk_idx, result.error or "Unknown error")

    def _create_trajectory_entry(self, results: list[ChunkResult]) -> dict:
        """Create a trajectory entry for the Root LM."""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return {
            "action": "parallel_extraction",
            "successful": successful,
            "failed": failed,
            "completion_rate": self.repl.get_completion_rate(),
        }

    def _should_finalize(self, results: list[ChunkResult]) -> bool:
        """Determine if we should finalize extraction."""
        # Finalize if all chunks processed successfully
        completion_rate = self.repl.get_completion_rate()
        success_count = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = success_count / max(total, 1)

        # Finalize if 100% complete OR if >80% success rate
        return completion_rate >= 1.0 or success_rate >= 0.8

    def _process_root_decision(
        self,
        task: str,
        trajectory: list[dict],
        yaml_schema: str,
        processor: ChunkProcessor,
        chunks: list[Chunk],
        logger: CallLogger,
    ) -> str:
        """Process Root LM decision for next action."""
        # Format trajectory for Root LM
        trajectory_str = self._format_trajectory(trajectory)

        # Get field completion summary
        field_completion = self.repl.get_field_completion_summary()

        # Log request
        root_lm = self.config.get_root_lm()
        request_payload = {
            "task": task,
            "trajectory": trajectory_str,
            "state_summary": self.repl.get_state_summary(),
            "chunk_summaries": self.repl.get_chunk_summaries_preview(),
            "results_preview": self.repl.get_results_preview(),
            "field_completion": field_completion,
            "retry_summary": self.repl.get_retry_summary(),
            "max_retries": self.config.max_retries,
        }
        call_id = logger.log_request(
            lm_type="root",
            model=str(root_lm),
            signature="RootExtractionSignature",
            call_type="root_decision",
            request=request_payload,
        )

        # Call Root LM
        with dspy.context(lm=root_lm):
            result = self.root_predictor(
                task=task,
                trajectory=trajectory_str,
                state_summary=self.repl.get_state_summary(),
                chunk_summaries=self.repl.get_chunk_summaries_preview(),
                results_preview=self.repl.get_results_preview(),
                field_completion=field_completion,
                retry_summary=self.repl.get_retry_summary(),
                max_retries=self.config.max_retries,
            )

        # Log response
        logger.log_response(
            call_id=call_id,
            response={
                "thought": getattr(result, "thought", ""),
                "action": getattr(result, "action", ""),
                "target_chunk": getattr(result, "target_chunk", ""),
                "targeted_prompt": getattr(result, "targeted_prompt", ""),
            },
        )

        # Parse action
        action = getattr(result, "action", "finalize").lower()

        if action == "finalize" or "finalize" in action:
            return "finalize"
        elif "re_extract" in action or "re-extract" in action:
            # Extract target chunk and prompt
            target_chunk = self._extract_target_chunk(result)
            targeted_prompt = getattr(result, "targeted_prompt", "")

            if target_chunk is not None and 0 <= target_chunk < len(chunks):
                # Re-extract specific chunk
                chunk_result = processor.process_chunk(
                    chunks[target_chunk],
                    yaml_schema,
                    targeted_prompt,
                )
                self._process_worker_results([chunk_result], chunks)

        return action

    def _format_trajectory(self, trajectory: list[dict]) -> str:
        """Format trajectory for Root LM prompt."""
        if not trajectory:
            return "No previous actions."

        lines = []
        for i, entry in enumerate(trajectory, 1):
            action = entry.get("action", "unknown")
            if action == "parallel_extraction":
                lines.append(
                    f"Turn {i}: Parallel extraction - "
                    f"{entry.get('successful', 0)} successful, "
                    f"{entry.get('failed', 0)} failed, "
                    f"{entry.get('completion_rate', 0):.1%} complete"
                )
            else:
                lines.append(f"Turn {i}: {action}")

        return "\n".join(lines)

    def _extract_target_chunk(self, result: dspy.Prediction) -> int | None:
        """Extract target chunk index from Root LM result."""
        target_str = getattr(result, "target_chunk", None)
        if target_str is None:
            return None

        try:
            return int(target_str)
        except (ValueError, TypeError):
            # Try to extract number from string
            import re

            match = re.search(r"\d+", str(target_str))
            return int(match.group()) if match else None

    def _compile_failures(self, chunks: list[Chunk]) -> list[dict]:
        """Compile failure information."""
        failures = []
        for idx, error in self.repl.failed_chunks.items():
            preview = ""
            if 0 <= idx < len(chunks):
                content = chunks[idx].content
                preview = content[:100] if len(content) > 100 else content

            failures.append(
                {
                    "chunk_idx": idx,
                    "error": error,
                    "content_preview": preview,
                }
            )
        return failures

    def _get_token_usage(self) -> dict:
        """Get token usage from DSPy."""
        # DSPy tracks usage when configured with track_usage=True
        # For now, return placeholder
        return {"root": 0, "worker": 0, "total": 0}
