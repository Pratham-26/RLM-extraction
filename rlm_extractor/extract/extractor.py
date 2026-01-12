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
from rlm_extractor.extract.chunker import Chunk, chunk_file, chunk_text
from rlm_extractor.extract.processor import ChunkProcessingResult as ChunkResult
from rlm_extractor.extract.processor import ChunkProcessor
from rlm_extractor.extract.schema import json_to_yaml, yaml_to_json
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

        # Initialize components
        self.chunk_size = config.chunk_size
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
        worker_lm = self.config.worker_lm

        # Initialize logger
        logger = CallLogger()

        # Convert schema (use compact mode if configured for efficiency)
        yaml_schema = json_to_yaml(json_schema, compact=self.config.compact_schema)

        # Chunk the document
        chunks = self._chunk_document(document)

        # Initialize REPL state
        input_context = document if isinstance(document, str) else [c.content for c in chunks]
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
            condensed_guidance = self._prepare_user_context(user_context, yaml_schema, logger)
            self.repl.set_condensed_guidance(condensed_guidance)

        # Create chunk processor
        processor = ChunkProcessor(
            worker_lm=worker_lm,
            max_parallel_workers=self.config.max_parallel_workers,
            condensed_guidance=condensed_guidance,
            logger=logger,
            chunk_timeout=self.config.chunk_timeout,
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

        # Convert entity contexts to final JSON
        final_json = self._convert_entity_contexts_to_json(json_schema, logger)

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
        """Validate input parameters."""
        # Validate json_schema
        if not json_schema or not isinstance(json_schema, dict) or "type" not in json_schema:
            raise ValueError("json_schema must be a non-empty dict with a 'type' field")

        # Validate document
        if document is None:
            raise ValueError("document cannot be None")
        if isinstance(document, str) and not document.strip():
            raise ValueError("document string cannot be empty")
        if isinstance(document, list) and not document:
            raise ValueError("document list cannot be empty")

    def _prepare_user_context(self, user_context: str, yaml_schema: str, logger: CallLogger) -> str:
        """Validate, sanitize, and condense user context."""
        # Validate
        if not isinstance(user_context, str):
            raise TypeError("user_context must be a string")

        if len(user_context) < MIN_USER_CONTEXT_CHARS:
            raise ValueError(f"user_context is too short (minimum {MIN_USER_CONTEXT_CHARS} chars)")

        max_chars = self.config.max_user_context_chars
        if len(user_context) > max_chars:
            raise ValueError(f"user_context is too long (maximum {max_chars} chars)")

        # Sanitize: remove control chars and limit repeated newlines
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", user_context)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

        # Condense with Root LM
        from rlm_extractor.signatures import ContextCondensationSignature

        condenser = dspy.Predict(ContextCondensationSignature)
        root_lm = self.config.root_lm

        call_id = logger.log_request(
            lm_type="root",
            model=str(root_lm),
            signature="ContextCondensationSignature",
            call_type="context_condensation",
            request={"user_context": sanitized, "yaml_schema": yaml_schema},
        )

        with dspy.context(lm=root_lm):
            result = condenser(user_context=sanitized, yaml_schema=yaml_schema)

        logger.log_response(
            call_id=call_id,
            response={"condensed_guidance": getattr(result, "condensed_guidance", "")},
        )

        return getattr(result, "condensed_guidance", "")

    def _chunk_document(
        self,
        document: str | list[str],
    ) -> list:
        """Chunk the document for processing."""
        if isinstance(document, str):
            # Check if it's a file path (exists or reasonable path length)
            is_file = (
                document
                and len(document) <= 1024
                and (os.path.exists(document) or os.path.exists(os.path.abspath(document)))
            )
            if is_file:
                return chunk_file(document, self.chunk_size)
            return chunk_text(document, self.chunk_size)
        else:
            # List of file paths
            chunks = []
            next_idx = 0
            for path in document:
                file_chunks = chunk_file(path, self.chunk_size)
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
                    confidence=result.confidence or "medium",
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
        # Format trajectory inline
        if trajectory:
            trajectory_lines = []
            for i, entry in enumerate(trajectory, 1):
                action = entry.get("action", "unknown")
                if action == "parallel_extraction":
                    trajectory_lines.append(
                        f"Turn {i}: Parallel extraction - "
                        f"{entry.get('successful', 0)} successful, "
                        f"{entry.get('failed', 0)} failed"
                    )
                else:
                    trajectory_lines.append(f"Turn {i}: {action}")
            trajectory_str = "\n".join(trajectory_lines)
        else:
            trajectory_str = "No previous actions."

        # Get field completion summary
        field_completion = self.repl.get_field_completion_summary()

        # Log request
        root_lm = self.config.root_lm
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
            # Extract target chunk inline
            target_str = getattr(result, "target_chunk", None)
            target_chunk = None
            if target_str is not None:
                try:
                    target_chunk = int(target_str)
                except (ValueError, TypeError):
                    match = re.search(r"\d+", str(target_str))
                    target_chunk = int(match.group()) if match else None
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

    def _convert_entity_contexts_to_json(self, json_schema: dict, logger: CallLogger) -> dict:
        """Convert accumulated entity contexts to structured JSON.

        This method uses the Root LM to parse the natural language entity
        contexts and extract the actual structured values matching the schema.

        Args:
            json_schema: The JSON Schema defining expected output structure
            logger: Call logger for tracking LM calls

        Returns:
            Dictionary with extracted values matching the schema
        """
        import json

        # If we have entity contexts, use Root LM to extract values
        if self.repl.entity_contexts:
            return self._extract_values_from_contexts(json_schema, logger)

        # Otherwise, fall back to results_so_far
        return yaml_to_json(
            json.dumps(self.repl.results_so_far),
            json_schema,
        )

    def _extract_values_from_contexts(self, json_schema: dict, logger: CallLogger) -> dict:
        """Use Root LM to extract structured values from entity contexts.

        Args:
            json_schema: The JSON Schema defining expected output structure
            logger: Call logger for tracking LM calls

        Returns:
            Dictionary with extracted values matching the schema
        """
        import json

        from rlm_extractor.signatures import RootValueExtractionSignature

        # Format entity contexts for the Root LM
        entity_contexts_str = self._format_entity_contexts_for_lm()

        # Get chunk summaries for context
        chunk_summaries_str = "\n".join([
            f"Chunk {s['idx']}: {s['gist']}"
            for s in self.repl.chunk_summaries
        ])

        # Get field completion status
        field_completion = self.repl.get_field_completion_summary()

        # Log request
        root_lm = self.config.root_lm
        call_id = logger.log_request(
            lm_type="root",
            model=str(root_lm),
            signature="RootValueExtractionSignature",
            call_type="value_extraction",
            request={
                "entity_contexts": entity_contexts_str[:500],
                "chunk_summaries": chunk_summaries_str[:200],
                "field_completion": field_completion,
            },
        )

        # Create predictor and call Root LM
        predictor = dspy.Predict(RootValueExtractionSignature)

        with dspy.context(lm=root_lm):
            result = predictor(
                yaml_schema=json_to_yaml(json_schema, compact=self.config.compact_schema),
                entity_contexts_all=entity_contexts_str,
                chunk_summaries=chunk_summaries_str,
                field_completion=field_completion,
            )

        # Log response
        extracted_values = getattr(result, "extracted_values", "")
        logger.log_response(
            call_id=call_id,
            response={
                "extracted_values": extracted_values[:500],
                "extraction_notes": getattr(result, "extraction_notes", ""),
            },
        )

        # Parse the YAML output back to JSON
        return self._parse_extracted_values_to_json(extracted_values, json_schema)

    def _format_entity_contexts_for_lm(self) -> str:
        """Format entity contexts for Root LM consumption.

        Returns:
            Formatted string with all entity contexts
        """
        if not self.repl.entity_contexts:
            return "No entity contexts collected."

        lines = ["Entity Contexts from all chunks:", ""]

        for field_name in sorted(self.repl.entity_contexts.keys()):
            contexts = self.repl.entity_contexts[field_name]
            lines.append(f"{field_name}:")

            for chunk_idx, description, conf in contexts:
                lines.append(f"  - [Chunk {chunk_idx}, confidence={conf}] {description}")

            lines.append("")  # Blank line between fields

        return "\n".join(lines)

    def _parse_extracted_values_to_json(self, yaml_str: str, json_schema: dict) -> dict:
        """Parse extracted values from YAML string to JSON.

        Args:
            yaml_str: YAML string from Root LM
            json_schema: JSON Schema for validation

        Returns:
            Dictionary matching the schema
        """
        import json
        import re

        if not yaml_str or yaml_str.strip() in ("", "none", "null"):
            return {}

        # Try to parse as YAML first
        try:
            import yaml
            parsed = yaml.safe_load(yaml_str)
            if isinstance(parsed, dict):
                return yaml_to_json(json.dumps(parsed), json_schema)
        except ImportError:
            pass
        except Exception:
            pass

        # Try JSON
        try:
            parsed = json.loads(yaml_str)
            if isinstance(parsed, dict):
                return yaml_to_json(json.dumps(parsed), json_schema)
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", yaml_str, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return yaml_to_json(json.dumps(parsed), json_schema)
            except json.JSONDecodeError:
                pass

        # If all parsing fails, return empty with the raw output for debugging
        return {"_raw_output": yaml_str, "_parsing_error": "Could not parse output as JSON/YAML"}
