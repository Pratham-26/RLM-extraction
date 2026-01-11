"""Tests for entity context extraction and parsing."""

import pytest

from rlm.extract.processor import ChunkProcessor
from rlm.repl import REPLState


class TestParseEntityContexts:
    """Test entity context parsing."""

    def test_parse_simple_contexts(self):
        """Test parsing simple field: description pairs."""
        processor = ChunkProcessor(worker_lm=None)

        contexts_str = """
        customer_name: Jane Smith is listed as the primary account holder in the customer info section
        invoice_number: INV-2024-12345 appears at the top of the document
        """

        result = processor._parse_entity_contexts(contexts_str)

        assert "customer_name" in result
        assert "invoice_number" in result
        assert "Jane Smith" in result["customer_name"]
        assert "INV-2024-12345" in result["invoice_number"]

    def test_parse_list_item_contexts(self):
        """Test parsing list items with _0, _1 suffixes."""
        processor = ChunkProcessor(worker_lm=None)

        contexts_str = """
        line_items_0: Widget A at $50 is the first item
        line_items_1: Widget B at $75 is the second item
        line_items_2: Widget C at $100 is the third item
        """

        result = processor._parse_entity_contexts(contexts_str)

        assert "line_items_0" in result
        assert "line_items_1" in result
        assert "line_items_2" in result

    def test_parse_empty_contexts(self):
        """Test parsing empty or null contexts."""
        processor = ChunkProcessor(worker_lm=None)

        assert processor._parse_entity_contexts("") == {}
        assert processor._parse_entity_contexts("none") == {}
        assert processor._parse_entity_contexts("null") == {}

    def test_parse_skips_llm_artifacts(self):
        """Test that common LLM artifacts are skipped."""
        processor = ChunkProcessor(worker_lm=None)

        contexts_str = """
        Entity Contexts:
        customer_name: Jane Smith
        Contexts:
        invoice_number: INV-12345
        """

        result = processor._parse_entity_contexts(contexts_str)

        assert "customer_name" in result
        assert "invoice_number" in result
        # The artifact headers should not create keys
        assert "Entity Contexts" not in result

    def test_parse_multiline_descriptions(self):
        """Test parsing with colons in descriptions."""
        processor = ChunkProcessor(worker_lm=None)

        contexts_str = """
        customer_name: Jane Smith: listed as primary account holder
        """

        result = processor._parse_entity_contexts(contexts_str)

        assert "customer_name" in result
        # Should split on first colon only
        assert "Jane Smith: listed as primary account holder" == result["customer_name"]


class TestREPLStateEntityContexts:
    """Test REPL state management of entity contexts."""

    def test_accumulate_contexts(self):
        """Test accumulating contexts from multiple chunks."""
        state = REPLState()

        # Chunk 0
        state.accumulate_entity_contexts(
            0,
            {"customer_name": "Jane Smith in header", "invoice_number": "INV-123"},
            "high"
        )

        # Chunk 1
        state.accumulate_entity_contexts(
            1,
            {"customer_name": "J. Smith in footer", "total": "$500"},
            "medium"
        )

        assert len(state.entity_contexts["customer_name"]) == 2
        assert state.entity_contexts["customer_name"][0] == (0, "Jane Smith in header", "high")
        assert state.entity_contexts["customer_name"][1] == (1, "J. Smith in footer", "medium")

        assert len(state.entity_contexts["invoice_number"]) == 1
        assert state.entity_contexts["invoice_number"][0] == (0, "INV-123", "high")

        assert len(state.entity_contexts["total"]) == 1
        assert state.entity_contexts["total"][0] == (1, "$500", "medium")

    def test_format_contexts_for_root(self):
        """Test formatting contexts for Root LM."""
        state = REPLState()

        state.accumulate_entity_contexts(
            0,
            {"field1": "Description 1", "field2": "Description 2"},
            "high"
        )
        state.accumulate_entity_contexts(
            1,
            {"field1": "Description 3"},
            "medium"
        )

        formatted = state.get_entity_contexts_for_root()

        assert "Entity Contexts from all chunks:" in formatted
        assert "field1:" in formatted
        assert "[Chunk 0, confidence=high]" in formatted
        assert "Description 1" in formatted
        assert "[Chunk 1, confidence=medium]" in formatted
        assert "Description 3" in formatted

    def test_format_empty_contexts(self):
        """Test formatting when no contexts collected."""
        state = REPLState()

        formatted = state.get_entity_contexts_for_root()

        assert formatted == "No entity contexts collected yet."

    def test_clear_on_reset(self):
        """Test that entity_contexts are cleared on reset."""
        state = REPLState()

        state.accumulate_entity_contexts(
            0,
            {"field1": "Description 1"},
            "high"
        )

        assert len(state.entity_contexts) > 0

        state.reset_for_task("test document", "yaml_schema")

        assert len(state.entity_contexts) == 0


class TestIsEntityContexts:
    """Test detection of entity contexts vs structured data."""

    def test_detects_entity_contexts(self):
        """Test that string descriptions are detected as contexts."""
        state = REPLState()

        # All values are strings longer than 20 chars -> contexts
        extracted = {
            "customer_name": "Jane Smith is listed as the primary account holder in this section",
            "email": "The email jane@example.com appears next to the customer name",
        }

        assert state._is_entity_contexts(extracted) is True

    def test_detects_structured_data(self):
        """Test that actual values are not detected as contexts."""
        state = REPLState()

        # Short values -> actual data, not contexts
        extracted = {
            "customer_name": "Jane Smith",
            "email": "jane@example.com",
        }

        assert state._is_entity_contexts(extracted) is False

    def test_detects_mixed_types(self):
        """Test that non-string values are not contexts."""
        state = REPLState()

        extracted = {
            "name": "Jane Smith",
            "age": 30,
            "active": True,
        }

        assert state._is_entity_contexts(extracted) is False

    def test_detects_empty_dict(self):
        """Test that empty dict returns False."""
        state = REPLState()

        assert state._is_entity_contexts({}) is False
