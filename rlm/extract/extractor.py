"""RLMExtractor - Main orchestrator for schema-based extraction.

Coordinates the entire extraction process:
1. Converts JSON Schema to YAML
2. Chunks the document
3. Runs parallel first-pass extraction
4. Orchestrates sequential re-extraction as needed
5. Aggregates results and converts back to JSON
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Union

import dspy
from PIL import Image

from rlm.config import RLMConfig
from rlm.extract.chunker import Chunker
from rlm.extract.processor import ChunkProcessor, ExtractionResult as ChunkResult
from rlm.extract.schema import SchemaConverter
from rlm.repl import REPLState
from rlm.signatures import RootExtractionSignature


@dataclass
class ExtractionResult:
    """Result of RLM schema extraction."""

    data: dict
    """Final extracted data matching the input JSON Schema."""

    chunk_gists: list[dict] = field(default_factory=list)
    """Summary of each chunk: [{idx, gist, confidence, fields_found}]."""

    failures: list[dict] = field(default_factory=list)
    """Failed chunks: [{chunk_idx, error, content_preview}]."""

    turns: int = 0
    """Total number of RLM orchestration turns."""

    token_usage: dict = field(default_factory=dict)
    """Token consumption: {root: int, worker: int, total: int}."""

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
            worker_vision_model="openai/gpt-4o",
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
        document: Union[str, list[Image.Image], list[str]],
        task: str | None = None,
    ) -> ExtractionResult:
        """Extract structured data from a document according to JSON Schema.

        Args:
            json_schema: JSON Schema defining what to extract
            document: Text string, list of PIL Images, or list of image paths
            task: Optional custom task description

        Returns:
            ExtractionResult with data, gists, failures, and usage
        """
        # Detect input modality
        modality = self._detect_modality(document)

        # Get appropriate worker LM
        worker_lm = self.config.get_worker_lm(modality)

        # Convert schema
        yaml_schema = self.schema_converter.json_to_yaml_chunks(json_schema)

        # Chunk the document
        chunks = self._chunk_document(document, modality)

        # Initialize REPL state
        self.repl.reset_for_task(
            input_context=document if isinstance(document, str) else [c.content for c in chunks],
            yaml_schema=yaml_schema,
        )
        self.repl.set_total_chunks(len(chunks))

        # Create chunk processor
        processor = ChunkProcessor(
            worker_lm=worker_lm,
            max_parallel_workers=self.config.max_parallel_workers,
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
            else:
                # Subsequent turns handled by root LM decisions
                action_result = self._process_root_decision(
                    task, trajectory, yaml_schema, processor, chunks
                )
                if action_result == "finalize":
                    break
                continue

            # Process results
            self._process_worker_results(results, chunks)
            trajectory.append(self._create_trajectory_entry(results))

            # Check if we should finalize
            if self._should_finalize(results):
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

        return ExtractionResult(
            data=final_json,
            chunk_gists=self.repl.chunk_summaries,
            failures=failures,
            turns=turn + 1,
            token_usage=usage,
        )

    def _detect_modality(self, document) -> str:
        """Determine if we need vision workers."""
        if isinstance(document, str):
            return "text"
        elif isinstance(document, list):
            if document and isinstance(document[0], Image.Image):
                return "vision"
            elif document and isinstance(document[0], str):
                # Could be image paths - assume vision
                return "vision"
        return "text"  # default

    def _chunk_document(
        self,
        document: Union[str, list[Image.Image], list[str]],
        modality: str,
    ) -> list:
        """Chunk the document for processing."""
        if modality == "text":
            return self.chunker.chunk_text(document)
        else:
            # Convert to list of Images if needed
            if isinstance(document, list) and document and isinstance(document[0], str):
                # Image paths - load them
                return self.chunker.chunk_image_files(document)
            else:
                # Already PIL Images
                return self.chunker.encode_images(document)

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
                    fields_found=result.missing_fields or [],
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
        # All chunks processed and acceptable success rate
        completion_rate = self.repl.get_completion_rate()
        failure_rate = len(self.repl.failed_chunks) / max(self.repl.total_chunks, 1)

        return completion_rate >= 1.0 or failure_rate < 0.1

    def _process_root_decision(
        self,
        task: str,
        trajectory: list,
        yaml_schema: str,
        processor: ChunkProcessor,
        chunks: list,
    ) -> str:
        """Process Root LM decision for next action."""
        # Format trajectory for Root LM
        trajectory_str = self._format_trajectory(trajectory)

        # Call Root LM
        with dspy.context(lm=self.config.get_root_lm()):
            result = self.root_predictor(
                task=task,
                trajectory=trajectory_str,
                state_summary=self.repl.get_state_summary(),
                chunk_summaries=self.repl.get_chunk_summaries_preview(),
                results_preview=self.repl.get_results_preview(),
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

    def _compile_failures(self, chunks: list) -> list[dict]:
        """Compile failure information."""
        failures = []
        for idx, error in self.repl.failed_chunks.items():
            preview = ""
            if 0 <= idx < len(chunks):
                content = chunks[idx].content
                preview = content[:100] if len(content) > 100 else content

            failures.append({
                "chunk_idx": idx,
                "error": error,
                "content_preview": preview,
            })
        return failures

    def _get_token_usage(self) -> dict:
        """Get token usage from DSPy."""
        # DSPy tracks usage when configured with track_usage=True
        # For now, return placeholder
        return {"root": 0, "worker": 0, "total": 0}
