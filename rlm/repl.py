"""REPL State Management for RLM.

The REPLState maintains persistent state across RLM turns, including:
- INPUT: The document (text or list of base64 images)
- Chunk tracking: completed, pending, failed
- Extraction results accumulated so far
- Chunk summaries for Root LM visibility
"""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class REPLState:
    """Persistent state across RLM extraction turns.

    The Root LM never sees the full INPUT - it only sees summaries and
    accumulated results. Workers see individual chunks.
    """

    # The document (never seen in full by Root LM)
    INPUT: Union[str, list[str]] = ""
    """Document as string (text mode) or list of base64 images (image mode)."""

    # Chunk tracking
    total_chunks: int = 0
    """Total number of chunks in the document."""

    completed_chunks: set[int] = field(default_factory=set)
    """Indices of chunks that have been processed successfully."""

    failed_chunks: dict[int, str] = field(default_factory=dict)
    """{chunk_idx: error_message} for failed chunks."""

    # What Root LM sees (not raw INPUT)
    chunk_summaries: list[dict] = field(default_factory=list)
    """[{idx, gist, confidence, fields_found}] for each processed chunk."""

    results_so_far: dict = field(default_factory=dict)
    """Accumulated extracted data in YAML/merged format."""

    # Configuration
    yaml_schema: str = ""
    """Full YAML schema for workers (not seen by Root LM in raw form)."""

    summary_level: str = "standard"
    """Detail level for gists: minimal/standard/verbose."""

    def get_pending_chunks(self) -> list[int]:
        """Return indices of chunks not yet processed."""
        return [i for i in range(self.total_chunks) if i not in self.completed_chunks]

    def get_completion_rate(self) -> float:
        """Return fraction of chunks completed (0.0 to 1.0)."""
        if self.total_chunks == 0:
            return 0.0
        return len(self.completed_chunks) / self.total_chunks

    def update_chunk_result(
        self,
        idx: int,
        gist: str,
        extracted: dict,
        confidence: str = "medium",
        fields_found: list[str] | None = None,
    ) -> None:
        """Called after worker completes successfully.

        Args:
            idx: Chunk index
            gist: Summary of chunk content
            extracted: Extracted data from this chunk
            confidence: high/medium/low
            fields_found: List of schema fields found in this chunk
        """
        self.completed_chunks.add(idx)

        # Store summary for Root LM
        self.chunk_summaries.append(
            {
                "idx": idx,
                "gist": gist,
                "confidence": confidence,
                "fields_found": fields_found or [],
            }
        )

        # Merge extracted data
        self._merge_extracted(extracted)

    def _merge_extracted(self, extracted: dict) -> None:
        """Merge new extraction into results_so_far.

        Handles:
        - Nested dictionaries (deep merge)
        - Lists (extend or append based on keys)
        - Scalar values (last write wins, with warning)
        """
        if not extracted:
            return

        for key, value in extracted.items():
            if key not in self.results_so_far:
                self.results_so_far[key] = value
            elif isinstance(value, dict) and isinstance(self.results_so_far[key], dict):
                # Deep merge dictionaries
                self._deep_merge(self.results_so_far[key], value)
            elif isinstance(value, list) and isinstance(self.results_so_far[key], list):
                # Extend lists
                self.results_so_far[key].extend(value)
            else:
                # Scalar or type mismatch - last write wins
                self.results_so_far[key] = value

    def _deep_merge(self, target: dict, source: dict) -> None:
        """Deep merge source into target dictionary."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            elif isinstance(value, list) and isinstance(target[key], list):
                target[key].extend(value)
            else:
                target[key] = value

    def mark_failed(self, idx: int, error: str) -> None:
        """Record a chunk failure."""
        self.failed_chunks[idx] = error

    def get_state_summary(self) -> str:
        """Return a formatted summary for Root LM."""
        pending = len(self.get_pending_chunks())
        failed = len(self.failed_chunks)

        summary_parts = [
            f"Completed: {len(self.completed_chunks)}/{self.total_chunks} chunks",
        ]

        if pending > 0:
            summary_parts.append(f"Pending: {pending}")

        if failed > 0:
            summary_parts.append(f"Failed: {failed}")

        return ", ".join(summary_parts)

    def get_results_preview(self, max_chars: int = 500) -> str:
        """Return a preview of results for Root LM."""
        if not self.results_so_far:
            return "No results yet."

        import json

        preview = json.dumps(self.results_so_far, indent=2)
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "\n... (truncated)"
        return preview

    def get_chunk_summaries_preview(self, max_count: int = 10) -> str:
        """Return a preview of chunk summaries for Root LM."""
        if not self.chunk_summaries:
            return "No chunks processed yet."

        lines = []
        for i, summary in enumerate(self.chunk_summaries[:max_count]):
            lines.append(
                f"Chunk {summary['idx']}: {summary['gist']} "
                f"(confidence: {summary['confidence']})"
            )

        if len(self.chunk_summaries) > max_count:
            lines.append(f"... and {len(self.chunk_summaries) - max_count} more chunks")

        return "\n".join(lines)

    def reset_for_task(self, input_context: Union[str, list[str]], yaml_schema: str) -> None:
        """Reset state for a new extraction task."""
        self.INPUT = input_context
        self.yaml_schema = yaml_schema

        if isinstance(input_context, str):
            # Text mode - chunks will be calculated externally
            self.total_chunks = 0
        else:
            # Image mode - one chunk per image
            self.total_chunks = len(input_context)

        self.completed_chunks.clear()
        self.failed_chunks.clear()
        self.chunk_summaries.clear()
        self.results_so_far.clear()

    def set_total_chunks(self, count: int) -> None:
        """Set the total number of chunks (for text mode)."""
        self.total_chunks = count
