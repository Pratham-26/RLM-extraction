"""Tests for field tracking in REPLState."""

import pytest

from rlm.repl import REPLState


class TestFieldTracking:
    """Test field tracking functionality."""

    def test_extract_required_fields_simple(self):
        """Test extracting required fields from simple schema."""
        state = REPLState()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }

        state.set_json_schema(schema)

        assert state.required_fields == {"name", "email"}

    def test_extract_required_fields_none(self):
        """Test schema with no required fields."""
        state = REPLState()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }

        state.set_json_schema(schema)

        assert state.required_fields == set()

    def test_extract_required_fields_empty(self):
        """Test empty required array."""
        state = REPLState()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": []
        }

        state.set_json_schema(schema)

        assert state.required_fields == set()

    def test_extract_required_fields_nested(self):
        """Test extracting required fields from nested schema."""
        state = REPLState()

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"}
                            }
                        }
                    },
                    "required": ["name", "address"]
                }
            },
            "required": ["person"]
        }

        state.set_json_schema(schema)

        # Should track nested required fields
        assert "person" in state.required_fields
        assert "person.name" in state.required_fields
        assert "person.address" in state.required_fields

    def test_extract_required_fields_array_of_objects(self):
        """Test extracting required fields from array of objects."""
        state = REPLState()

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"}
                        },
                        "required": ["name", "price"]
                    }
                }
            },
            "required": ["items"]
        }

        state.set_json_schema(schema)

        # Should track array item required fields
        assert "items" in state.required_fields
        assert "items[].name" in state.required_fields
        assert "items[].price" in state.required_fields

    def test_extract_required_fields_invalid_schema(self):
        """Test handling of invalid schema gracefully."""
        state = REPLState()

        # Not an object type
        state.set_json_schema({"type": "string"})

        assert state.required_fields == set()

    def test_update_fields_found(self):
        """Test updating fields found across chunks."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"]
        })

        state.update_fields_found(["a", "x"])
        state.update_fields_found(["b", "y"])

        assert state.fields_found_all == {"a", "b", "x", "y"}
        assert state.required_fields_found == {"a", "b"}

    def test_update_fields_found_empty(self):
        """Test updating with empty list."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {"a": {"type": "string"}}
        })

        state.update_fields_found([])

        assert state.fields_found_all == set()
        assert state.required_fields_found == set()

    def test_get_missing_required_fields(self):
        """Test getting missing required fields."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total": {"type": "number"},
                "date": {"type": "string"}
            },
            "required": ["invoice_number", "total", "date"]
        })

        state.update_fields_found(["invoice_number"])

        missing = state.get_missing_required_fields()
        assert set(missing) == {"date", "total"}

    def test_get_missing_required_fields_all_found(self):
        """Test when all required fields are found."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a", "b"]
        })

        state.update_fields_found(["a", "b"])

        missing = state.get_missing_required_fields()
        assert missing == []

    def test_get_missing_required_fields_none_required(self):
        """Test when no required fields defined."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            }
        })

        state.update_fields_found(["a"])

        missing = state.get_missing_required_fields()
        assert missing == []

    def test_get_field_completion_summary_with_required(self):
        """Test field completion summary with required fields."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b", "c"]
        })

        state.update_fields_found(["a"])

        summary = state.get_field_completion_summary()

        assert "Required fields: 1/3 found" in summary
        assert "Missing required" in summary
        assert "b" in summary
        assert "c" in summary

    def test_get_field_completion_summary_all_required_found(self):
        """Test field completion summary when all required found."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a", "b"]
        })

        state.update_fields_found(["a", "b"])

        summary = state.get_field_completion_summary()

        assert "Required fields: 2/2 found" in summary
        assert "Missing required" not in summary

    def test_get_field_completion_summary_no_required(self):
        """Test field completion summary without required fields."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            }
        })

        state.update_fields_found(["a", "b"])

        summary = state.get_field_completion_summary()

        assert "Fields found: 2" in summary

    def test_get_field_completion_summary_many_missing(self):
        """Test field completion summary truncates long missing list."""
        state = REPLState()
        # Create schema with 10 required fields
        properties = {f"field_{i}": {"type": "string"} for i in range(10)}
        state.set_json_schema({
            "type": "object",
            "properties": properties,
            "required": list(properties.keys())
        })

        # Only find first 2
        state.update_fields_found(["field_0", "field_1"])

        summary = state.get_field_completion_summary()

        assert "Required fields: 2/10 found" in summary
        assert "Missing required:" in summary
        assert "..." in summary  # Should have truncation indicator

    def test_update_chunk_result_tracks_fields(self):
        """Test that update_chunk_result updates field tracking."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        })

        state.update_chunk_result(
            idx=0,
            gist="Test",
            extracted={"name": "John"},
            confidence="high",
            fields_found=["name"]
        )

        assert "name" in state.fields_found_all
        assert "name" in state.required_fields_found

    def test_update_chunk_result_with_empty_fields_found(self):
        """Test update_chunk_result with empty fields_found."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        })

        state.update_chunk_result(
            idx=0,
            gist="Test",
            extracted={},
            confidence="low",
            fields_found=[]
        )

        assert state.fields_found_all == set()
        assert state.required_fields_found == set()

    def test_reset_clears_field_tracking(self):
        """Test that reset_for_task clears field tracking."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "a": {"type": "string"}
            },
            "required": ["a"]
        })
        state.update_fields_found(["a"])

        state.reset_for_task("new input", "new schema")

        # Field tracking should be cleared
        assert state.fields_found_all == set()
        assert state.required_fields_found == set()
        # But required_fields should persist (from set_json_schema)
        assert state.required_fields == {"a"}

    def test_backward_compatibility_no_schema_set(self):
        """Test backward compatibility when json_schema is not set."""
        state = REPLState()
        # Don't set json_schema

        state.update_fields_found(["a", "b"])

        # Should not error
        assert state.fields_found_all == {"a", "b"}
        assert state.get_field_completion_summary() == "Fields found: 2"

    def test_multiple_chunk_updates(self):
        """Test field tracking across multiple chunks."""
        state = REPLState()
        state.set_json_schema({
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total": {"type": "number"},
                "date": {"type": "string"},
                "vendor": {"type": "string"}
            },
            "required": ["invoice_number", "total", "date"]
        })

        # Chunk 0 finds invoice_number
        state.update_fields_found(["invoice_number"])
        # Chunk 1 finds total
        state.update_fields_found(["total"])
        # Chunk 2 finds nothing new
        state.update_fields_found([])
        # Chunk 3 finds vendor (not required)
        state.update_fields_found(["vendor"])

        assert state.fields_found_all == {"invoice_number", "total", "vendor"}
        assert state.required_fields_found == {"invoice_number", "total"}
        assert state.get_missing_required_fields() == ["date"]

    def test_set_json_schema_overwrites_previous(self):
        """Test that set_json_schema overwrites previous required fields."""
        state = REPLState()

        # First schema
        state.set_json_schema({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"]
        })
        assert state.required_fields == {"a"}

        # Second schema - should overwrite
        state.set_json_schema({
            "type": "object",
            "properties": {"b": {"type": "string"}},
            "required": ["b"]
        })
        assert state.required_fields == {"b"}
        # Note: this doesn't clear required_fields_found since that's separate
