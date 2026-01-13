"""Context Compaction - Intelligent summarization for scaling entity contexts.

Provides hierarchical compaction strategies to prevent Root LM context from
growing linearly with chunk count while preserving information.

Strategies:
1. Direct pass: < threshold, no compaction
2. Confidence-based: Keep high-confidence, summarize low-confidence
3. Semantic clustering: Group similar contexts, keep representatives
4. Aggregate summaries: Statistical summaries for very large sets
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass
class EntityContext:
    """A single entity context with metadata."""
    chunk_idx: int
    description: str
    confidence: str

    @property
    def confidence_score(self) -> int:
        """Convert confidence string to numeric score."""
        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(self.confidence.lower(), 2)


@dataclasses.dataclass
class CompactedContexts:
    """Result of context compaction."""
    formatted: str  # The formatted string for Root LM
    original_count: int  # Total contexts before compaction
    retained_count: int  # Contexts kept as-is
    summarized_count: int  # Contexts that were summarized
    aggregate_count: int  # Contexts represented in aggregate summaries


class ContextCompactor:
    """Intelligently compacts entity contexts to prevent unbounded growth."""

    def __init__(
        self,
        max_contexts: int = 50,
        enable_compaction: bool = True,
        confidence_threshold: str = "medium",
    ):
        """Initialize compactor.

        Args:
            max_contexts: Maximum contexts to retain per field
            enable_compaction: Use intelligent compaction (vs simple truncation)
            confidence_threshold: Minimum confidence to keep without summarization
        """
        self.max_contexts = max_contexts
        self.enable_compaction = enable_compaction
        self.confidence_threshold = confidence_threshold

    def compact_field_contexts(
        self,
        field_name: str,
        contexts: list[tuple[int, str, str]],
    ) -> CompactedContexts:
        """Compact contexts for a single field.

        Args:
            field_name: Name of the field
            contexts: List of (chunk_idx, description, confidence) tuples

        Returns:
            CompactedContexts with formatted string and metadata
        """
        if not contexts:
            return CompactedContexts(
                formatted=f"{field_name}: (no contexts)",
                original_count=0,
                retained_count=0,
                summarized_count=0,
                aggregate_count=0,
            )

        original_count = len(contexts)

        # Direct pass if under threshold
        if original_count <= self.max_contexts:
            formatted = self._format_all_contexts(field_name, contexts)
            return CompactedContexts(
                formatted=formatted,
                original_count=original_count,
                retained_count=original_count,
                summarized_count=0,
                aggregate_count=0,
            )

        # Apply compaction strategy
        if self.enable_compaction:
            return self._intelligent_compaction(field_name, contexts)
        else:
            return self._simple_truncation(field_name, contexts)

    def _format_all_contexts(
        self,
        field_name: str,
        contexts: list[tuple[int, str, str]],
    ) -> str:
        """Format all contexts without compaction."""
        lines = [f"{field_name}:"]
        for chunk_idx, description, conf in contexts:
            lines.append(f"  - [Chunk {chunk_idx}, confidence={conf}] {description}")
        return "\n".join(lines)

    def _simple_truncation(
        self,
        field_name: str,
        contexts: list[tuple[int, str, str]],
    ) -> CompactedContexts:
        """Simple truncation with count indicator."""
        lines = [f"{field_name}:"]
        retained = contexts[: self.max_contexts]
        truncated = len(contexts) - self.max_contexts

        for chunk_idx, description, conf in retained:
            lines.append(f"  - [Chunk {chunk_idx}, confidence={conf}] {description}")

        if truncated > 0:
            lines.append(f"  ... and {truncated} more contexts (truncated)")

        return CompactedContexts(
            formatted="\n".join(lines),
            original_count=len(contexts),
            retained_count=len(retained),
            summarized_count=0,
            aggregate_count=truncated,
        )

    def _intelligent_compaction(
        self,
        field_name: str,
        contexts: list[tuple[int, str, str]],
    ) -> CompactedContexts:
        """Apply intelligent compaction strategy.

        Strategy:
        1. Keep all high-confidence contexts
        2. Keep top medium-confidence contexts up to budget
        3. Summarize remaining contexts by similarity patterns
        """
        # Separate by confidence
        high_conf = [c for c in contexts if c[2].lower() == "high"]
        medium_conf = [c for c in contexts if c[2].lower() == "medium"]
        low_conf = [c for c in contexts if c[2].lower() == "low"]

        lines = [f"{field_name}:"]

        # Keep all high-confidence contexts
        retained = []
        retained.extend(high_conf)

        # Fill remaining budget with best medium-confidence
        remaining_budget = self.max_contexts - len(retained)
        if remaining_budget > 0:
            retained.extend(medium_conf[:remaining_budget])

        # Format retained contexts
        for chunk_idx, description, conf in retained:
            lines.append(f"  - [Chunk {chunk_idx}, confidence={conf}] {description}")

        # Summarize the rest
        omitted = len(contexts) - len(retained)
        if omitted > 0:
            summary = self._create_aggregate_summary(
                contexts[len(retained):],
                omitted,
            )
            lines.append(f"  - {summary}")

        return CompactedContexts(
            formatted="\n".join(lines),
            original_count=len(contexts),
            retained_count=len(retained),
            summarized_count=omitted if omitted > 0 else 0,
            aggregate_count=omitted if omitted > 0 else 0,
        )

    def _create_aggregate_summary(
        self,
        contexts: list[tuple[int, str, str]],
        count: int,
    ) -> str:
        """Create an aggregate summary for omitted contexts.

        Analyzes patterns in the omitted contexts to create a meaningful summary.
        """
        if not contexts:
            return f"{count} contexts omitted"

        # Extract descriptions
        descriptions = [c[1] for c in contexts]

        # Find common patterns (simple keyword-based approach)
        # For production, consider using embeddings for semantic clustering
        summary_parts = []

        # Check for common prefixes/starting words
        first_words = [d.split()[0].lower() for d in descriptions if d.split()]
        if first_words:
            from collections import Counter
            common = Counter(first_words).most_common(3)
            if common:
                patterns = [f"{word} ({cnt})" for word, cnt in common if cnt > 1]
                if patterns:
                    summary_parts.append(f"common themes: {', '.join(patterns)}")

        # Average length indicator
        avg_len = sum(len(d) for d in descriptions) / len(descriptions)
        if avg_len > 100:
            summary_parts.append("detailed descriptions")

        # Build summary
        base = f"+ {count} more contexts"
        if summary_parts:
            return f"{base} ({', '.join(summary_parts)})"
        return base


def compact_all_entity_contexts(
    entity_contexts: dict[str, list[tuple[int, str, str]]],
    max_contexts: int = 50,
    enable_compaction: bool = True,
) -> CompactedContexts:
    """Compact all entity contexts for Root LM consumption.

    Args:
        entity_contexts: Dict of {field_name: [(chunk_idx, description, conf), ...]}
        max_contexts: Maximum contexts per field
        enable_compaction: Use intelligent compaction

    Returns:
        CompactedContexts with formatted string for all fields
    """
    compactor = ContextCompactor(
        max_contexts=max_contexts,
        enable_compaction=enable_compaction,
    )

    lines = ["Entity Contexts from all chunks:", ""]
    total_original = 0
    total_retained = 0
    total_summarized = 0

    for field_name in sorted(entity_contexts.keys()):
        contexts = entity_contexts[field_name]
        total_original += len(contexts)

        result = compactor.compact_field_contexts(field_name, contexts)
        lines.append(result.formatted)
        lines.append("")  # Blank line between fields

        total_retained += result.retained_count
        total_summarized += result.summarized_count

    return CompactedContexts(
        formatted="\n".join(lines),
        original_count=total_original,
        retained_count=total_retained,
        summarized_count=total_summarized,
        aggregate_count=total_summarized,
    )
