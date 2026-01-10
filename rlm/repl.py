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
    # Document as string (text mode) or list of base64 images (image mode)
    INPUT: Union[str, list[str]] = ""

    # Chunk tracking
    # Total number of chunks in the document
    total_chunks: int = 0

    # Indices of chunks that have been processed successfully
    completed_chunks: set[int] = field(default_factory=set)

    # {chunk_idx: error_message} for failed chunks
    failed_chunks: dict[int, str] = field(default_factory=dict)

    # What Root LM sees (not raw INPUT)
    # [{idx, gist, confidence, fields_found}] for each processed chunk
    chunk_summaries: list[dict] = field(default_factory=list)

    # Accumulated extracted data in YAML/merged format
    results_so_far: dict = field(default_factory=dict)

    # Configuration
    # Full YAML schema for workers (not seen by Root LM in raw form)
    yaml_schema: str = ""

    # Detail level for gists: minimal/standard/verbose
    summary_level: str = "standard"

    # Condensed user guidance for workers (produced by Root LM from user_context)
    condensed_guidance: str = ""

    # Field tracking across all chunks
    # Original JSON Schema (stored to identify required fields)
    json_schema: dict = field(default_factory=dict)

    # All schema field names found across all chunks
    fields_found_all: set[str] = field(default_factory=set)

    # Required fields from the JSON Schema
    required_fields: set[str] = field(default_factory=set)

    # Required fields that have been found
    required_fields_found: set[str] = field(default_factory=set)

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

        # Track fields found
        if fields_found:
            self.update_fields_found(fields_found)

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

        # Clear field tracking for new task
        self.fields_found_all.clear()
        self.required_fields_found.clear()
        # Note: json_schema and required_fields persist until explicitly set

    def set_total_chunks(self, count: int) -> None:
        """Set the total number of chunks (for text mode)."""
        self.total_chunks = count

    def set_condensed_guidance(self, guidance: str) -> None:
        """Set the condensed user guidance for workers.

        Args:
            guidance: Condensed extraction guidance from Root LM
        """
        self.condensed_guidance = guidance

    def set_json_schema(self, json_schema: dict) -> None:
        """Store the original JSON Schema and extract required fields.

        Args:
            json_schema: Original JSON Schema dict
        """
        self.json_schema = json_schema
        self.required_fields.clear()  # Clear previous required fields
        self._extract_required_fields(json_schema)

    def _extract_required_fields(self, schema: dict, prefix: str = "") -> None:
        """Extract required field names from JSON Schema.

        Handles nested objects by building dot-notation paths.

        Args:
            schema: JSON Schema dict
            prefix: Current path prefix for nested fields
        """
        if not isinstance(schema, dict):
            return

        schema_type = schema.get("type")
        if schema_type != "object":
            return

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Track required fields at this level
        for prop_name in required:
            if prop_name in properties:
                full_name = f"{prefix}.{prop_name}" if prefix else prop_name
                self.required_fields.add(full_name)

        # Recursively handle nested objects
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue

            full_name = f"{prefix}.{prop_name}" if prefix else prop_name

            if prop_schema.get("type") == "object":
                self._extract_required_fields(prop_schema, full_name)
            elif prop_schema.get("type") == "array":
                items = prop_schema.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    # For arrays of objects, track with [] notation
                    self._extract_required_fields(items, f"{full_name}[]")

    def update_fields_found(self, fields_found: list[str]) -> None:
        """Update the aggregate set of fields found across all chunks.

        Args:
            fields_found: List of field names found in a chunk
        """
        for field in fields_found:
            self.fields_found_all.add(field)
            # Also track if this is a required field
            if field in self.required_fields:
                self.required_fields_found.add(field)

    def get_missing_required_fields(self) -> list[str]:
        """Return list of required fields not yet found.

        Returns:
            Sorted list of missing required field names
        """
        missing = self.required_fields - self.required_fields_found
        return sorted(list(missing))

    def get_field_completion_summary(self) -> str:
        """Return a formatted summary of field completion for Root LM.

        Returns:
            String describing field completion status
        """
        if not self.required_fields:
            # No required fields defined
            found_count = len(self.fields_found_all)
            return f"Fields found: {found_count}"

        required_total = len(self.required_fields)
        required_found = len(self.required_fields_found)
        required_missing = required_total - required_found

        parts = [f"Required fields: {required_found}/{required_total} found"]

        if required_missing > 0:
            missing = self.get_missing_required_fields()
            parts.append(f"Missing required: {', '.join(missing[:5])}")
            if len(missing) > 5:
                parts.append(f"... and {len(missing) - 5} more")

        return ". ".join(parts) + "."
