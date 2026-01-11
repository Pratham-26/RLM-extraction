"""Tests for SchemaConverter."""

import pytest

from rlm_extractor.extract.schema import SchemaConverter


class TestSchemaConverter:
    """Test schema conversion functionality."""

    def test_init(self):
        """Test converter initialization."""
        converter = SchemaConverter()
        assert converter is not None

    def test_json_to_yaml_simple(self):
        """Test converting simple JSON Schema to YAML."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "# Extraction Schema" in yaml_str
        assert "name:" in yaml_str
        assert "age:" in yaml_str
        assert "string" in yaml_str
        assert "integer" in yaml_str

    def test_json_to_yaml_nested(self):
        """Test converting nested JSON Schema to YAML."""
        converter = SchemaConverter()

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
                                "city": {"type": "string"},
                            },
                        },
                    },
                }
            },
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "person:" in yaml_str
        assert "name:" in yaml_str
        assert "address:" in yaml_str
        assert "street:" in yaml_str
        assert "city:" in yaml_str

    def test_json_to_yaml_array(self):
        """Test converting JSON Schema with arrays to YAML."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
                    },
                }
            },
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "items:" in yaml_str
        assert "Array" in yaml_str

    def test_json_to_yaml_with_descriptions(self):
        """Test converting schema with descriptions."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "User's email address",
                    "format": "email",
                }
            },
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "email" in yaml_str
        # Description text should be included as a comment
        assert "User's email address" in yaml_str or "user's email address" in yaml_str

    def test_json_to_yaml_required(self):
        """Test that required fields are marked."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "required_field" in yaml_str
        assert "optional_field" in yaml_str
        assert "(required)" in yaml_str
