"""Tests for SchemaConverter."""

import pytest

from rlm.extract.schema import SchemaConverter


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
            }
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
                                "city": {"type": "string"}
                            }
                        }
                    }
                }
            }
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
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"}
                        }
                    }
                }
            }
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
                    "format": "email"
                }
            }
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
                "optional_field": {"type": "string"}
            },
            "required": ["required_field"]
        }

        yaml_str = converter.json_to_yaml_chunks(schema)

        assert "required_field" in yaml_str
        assert "optional_field" in yaml_str
        assert "(required)" in yaml_str

    def test_yaml_to_json_simple(self):
        """Test converting YAML back to JSON."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            }
        }

        yaml_data = """
name: John Doe
age: "30"
active: "true"
"""

        result = converter.yaml_to_json(yaml_data, schema)

        assert result["name"] == "John Doe"
        assert result["age"] == 30  # Converted to int
        assert result["active"] is True  # Converted to bool

    def test_yaml_to_json_type_conversion(self):
        """Test type conversion in YAML to JSON."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "enabled": {"type": "boolean"}
            }
        }

        yaml_data = """
count: "42"
price: "19.99"
enabled: "yes"
"""

        result = converter.yaml_to_json(yaml_data, schema)

        assert result["count"] == 42
        assert result["price"] == 19.99
        assert result["enabled"] is True

    def test_yaml_to_json_nested(self):
        """Test converting nested YAML to JSON."""
        converter = SchemaConverter()

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }

        yaml_data = """
person:
  name: Jane
  age: "25"
"""

        result = converter.yaml_to_json(yaml_data, schema)

        assert result["person"]["name"] == "Jane"
        assert result["person"]["age"] == 25

    def test_yaml_to_json_arrays(self):
        """Test converting YAML arrays to JSON."""
        converter = SchemaConverter()

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
                        }
                    }
                }
            }
        }

        yaml_data = """
items:
  - name: Item 1
    price: "10.00"
  - name: Item 2
    price: "20.00"
"""

        result = converter.yaml_to_json(yaml_data, schema)

        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "Item 1"
        assert result["items"][0]["price"] == 10.0

    def test_fix_and_parse_yaml(self):
        """Test fixing malformed YAML."""
        converter = SchemaConverter()

        # Slightly malformed YAML
        malformed = "name: Test\nage: 30"

        result = converter._fix_and_parse_yaml(malformed)

        assert result["name"] == "Test"
        assert result["age"] == 30
