"""REPL State Management for RLM.

The REPLState maintains persistent state across RLM turns, including:
- INPUT: The document (text or list of base64 images)
- Chunk tracking: completed, pending, failed
- Extraction results accumulated so far
- Chunk summaries for Root LM visibility
"""

import threading
from dataclasses import dataclass, field


@dataclass
class REPLState:
    """Persistent state across RLM extraction turns.

    The Root LM never sees the full INPUT - it only sees summaries and
    accumulated results. Workers see individual chunks.

    Thread-safe: update_chunk_result() and mark_failed() are protected by a lock
    for use with parallel worker processing.
    """

    # Thread safety lock for parallel updates
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # The document (never seen in full by Root LM)
    # Document as string (text mode) or list of image placeholders (image mode)
    INPUT: str | list[str] = ""

    # Chunk tracking
    # Total number of chunks in the document
    total_chunks: int = 0

    # Indices of chunks that have been processed successfully
    completed_chunks: set[int] = field(default_factory=set)

    # {chunk_idx: error_message} for failed chunks
    failed_chunks: dict[int, str] = field(default_factory=dict)

    # {chunk_idx: retry_count} tracking how many times each chunk was processed
    chunk_retry_counts: dict[int, int] = field(default_factory=dict)

    # What Root LM sees (not raw INPUT)
    # [{idx, gist, confidence, fields_found}] for each processed chunk
    chunk_summaries: list[dict] = field(default_factory=list)

    # Accumulated data (entity contexts or structured values)
    results_so_far: dict = field(default_factory=dict)

    # Accumulated entity contexts: {field_name: [(chunk_idx, description, confidence), ...]}
    entity_contexts: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)

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

        Thread-safe - protected by lock for parallel processing.

        Args:
            idx: Chunk index
            gist: Summary of chunk content
            extracted: Extracted data from this chunk
            confidence: high/medium/low
            fields_found: List of schema fields found in this chunk
        """
        with self._lock:
            self.completed_chunks.add(idx)

            # Increment retry count for this chunk
            self.chunk_retry_counts[idx] = self.chunk_retry_counts.get(idx, 0) + 1

            # Track fields found
            if fields_found:
                self._update_fields_found_unsafe(fields_found)

            # Store summary for Root LM
            self.chunk_summaries.append(
                {
                    "idx": idx,
                    "gist": gist,
                    "confidence": confidence,
                    "fields_found": fields_found or [],
                }
            )

            # Store extracted data
            self._merge_extracted(extracted)

            # Also accumulate entity contexts if present
            if self._is_entity_contexts(extracted):
                self.accumulate_entity_contexts(idx, extracted, confidence)

    def _is_entity_contexts(self, extracted: dict) -> bool:
        """Check if extracted dict contains entity contexts vs structured data.

        Entity contexts have string values describing the entity.
        Structured data has the actual values with proper types.

        Args:
            extracted: Dict from worker result

        Returns:
            True if appears to be entity contexts
        """
        if not extracted:
            return False

        # Check if all values are strings (likely contexts)
        # Contexts tend to be longer than 30 chars
        for value in extracted.values():
            if not isinstance(value, str):
                return False
            if len(value) < 20:  # Likely a real value, not a description
                return False

        return True

    def accumulate_entity_contexts(
        self,
        idx: int,
        entity_contexts: dict[str, str],
        confidence: str = "medium",
    ) -> None:
        """Accumulate entity contexts from a chunk.

        Thread-safe - protected by lock for parallel processing.

        Args:
            idx: Chunk index
            entity_contexts: Dict of {field_name: context_description} from this chunk
            confidence: high/medium/low (used for conflict resolution)
        """
        with self._lock:
            for field_name, description in entity_contexts.items():
                if field_name not in self.entity_contexts:
                    self.entity_contexts[field_name] = []

                self.entity_contexts[field_name].append((idx, description, confidence))

    def get_entity_contexts_for_root(self, max_contexts: int = 50) -> str:
        """Format entity contexts for Root LM value extraction.

        Args:
            max_contexts: Maximum number of contexts to include per field

        Returns:
            Formatted string with all entity contexts
        """
        if not self.entity_contexts:
            return "No entity contexts collected yet."

        lines = ["Entity Contexts from all chunks:", ""]

        for field_name in sorted(self.entity_contexts.keys()):
            contexts = self.entity_contexts[field_name]
            lines.append(f"{field_name}:")

            for chunk_idx, description, conf in contexts[:max_contexts]:
                lines.append(f"  [Chunk {chunk_idx}, confidence={conf}] {description}")

            if len(contexts) > max_contexts:
                lines.append(f"  ... and {len(contexts) - max_contexts} more")

            lines.append("")  # Blank line between fields

        return "\n".join(lines)

    def _merge_extracted(self, extracted: dict) -> None:
        """Store extraction in results_so_far.

        Note: With entity contexts, this is now just a simple store.
        The actual value extraction happens in Root LM via _extract_values_from_contexts().
        """
        if not extracted:
            return
        # Simple store - entity contexts are accumulated separately
        for key, value in extracted.items():
            self.results_so_far[key] = value

    def mark_failed(self, idx: int, error: str) -> None:
        """Record a chunk failure.

        Thread-safe - protected by lock for parallel processing.
        """
        with self._lock:
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
            retry_count = self.chunk_retry_counts.get(summary["idx"], 0)
            retry_info = f" (retry {retry_count})" if retry_count > 0 else ""
            lines.append(
                f"Chunk {summary['idx']}: {summary['gist']} "
                f"(confidence: {summary['confidence']}){retry_info}"
            )

        if len(self.chunk_summaries) > max_count:
            lines.append(f"... and {len(self.chunk_summaries) - max_count} more chunks")

        return "\n".join(lines)

    def get_retry_summary(self) -> str:
        """Return a summary of retry counts for Root LM.

        Returns:
            String describing retry status for chunks with retries
        """
        if not self.chunk_retry_counts:
            return "No re-extractions yet."

        retried_chunks = {idx: count for idx, count in self.chunk_retry_counts.items() if count > 1}
        if not retried_chunks:
            return "No chunks re-extracted yet."

        parts = []
        for idx, count in sorted(retried_chunks.items()):
            parts.append(f"Chunk {idx}: {count} total attempts")

        return "Re-extractions: " + ", ".join(parts)

    def reset_for_task(self, input_context: str | list[str], yaml_schema: str) -> None:
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
        self.entity_contexts.clear()
        self.chunk_retry_counts.clear()

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

    def _update_fields_found_unsafe(self, fields_found: list[str]) -> None:
        """Update the aggregate set of fields found across all chunks.

        Not thread-safe - must be called with lock held.
        """
        for field in fields_found:
            self.fields_found_all.add(field)
            # Also track if this is a required field
            if field in self.required_fields:
                self.required_fields_found.add(field)

    def update_fields_found(self, fields_found: list[str]) -> None:
        """Update the aggregate set of fields found across all chunks.

        Thread-safe - protected by lock for parallel processing.

        Args:
            fields_found: List of field names found in a chunk
        """
        with self._lock:
            self._update_fields_found_unsafe(fields_found)

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
