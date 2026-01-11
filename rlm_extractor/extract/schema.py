"""Schema Converter - JSON Schema to YAML transformation.

Converts JSON Schema to YAML format optimized for LLM comprehension.
Uses pyyaml for all conversions - no LLM calls.
"""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class SchemaConverter:
    """Convert between JSON Schema and YAML for LM processing.

    Users provide JSON Schema; internally we use YAML for better LM comprehension.
    This is an implementation detail - users never see YAML.
    """

    def __init__(self) -> None:
        """Initialize converter."""
        pass

    def json_to_yaml_chunks(self, json_schema: dict) -> str:
        """Convert JSON Schema to YAML format optimized for LM comprehension.

        Args:
            json_schema: JSON Schema as a dict

        Returns:
            YAML string with type hints, descriptions, and clear structure
        """
        yaml_lines = []
        yaml_lines.append("# Extraction Schema")
        yaml_lines.append("# Extract the following fields from the document:\n")

        self._convert_schema_node(json_schema, yaml_lines, level=0)

        return "\n".join(yaml_lines)

    def _convert_schema_node(
        self,
        schema: dict,
        output: list[str],
        level: int = 0,
        name: str = "",
    ) -> None:
        """Recursively convert a schema node to YAML representation."""
        indent = "  " * level

        # Handle the schema itself (root object)
        if name:
            output.append(f"{indent}{name}:")

        schema_type = schema.get("type")
        description = schema.get("description", "")
        title = schema.get("title", "")

        # Add metadata as comments
        if title:
            output.append(f"{indent}  # Title: {title}")
        if description:
            output.append(f"{indent}  # Description: {description}")

        if schema_type == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for prop_name, prop_schema in properties.items():
                is_required = prop_name in required
                req_marker = " (required)" if is_required else " (optional)"

                # Get type info
                prop_type = prop_schema.get("type", "string")
                prop_desc = prop_schema.get("description", "")
                prop_pattern = prop_schema.get("pattern", "")
                prop_format = prop_schema.get("format", "")
                prop_enum = prop_schema.get("enum", [])

                # Comment line with type and constraints
                type_comment = f"{prop_type}"
                if prop_format:
                    type_comment += f" (format: {prop_format})"
                if prop_pattern:
                    type_comment += f" (pattern: {prop_pattern})"
                if prop_enum:
                    type_comment += f" (enum: {', '.join(map(str, prop_enum))})"

                output.append(f"{indent}  # {prop_name}: {type_comment}{req_marker}")
                if prop_desc:
                    output.append(f"{indent}  #   {prop_desc}")

                if prop_type == "object" and "properties" in prop_schema:
                    output.append(f"{indent}  {prop_name}:")
                    self._convert_schema_node(prop_schema, output, level + 2, "")
                elif prop_type == "array":
                    items = prop_schema.get("items", {})
                    output.append(f"{indent}  {prop_name}:")
                    if isinstance(items, dict) and "properties" in items:
                        # Array of objects
                        output.append(f"{indent}    - # Array of objects with:")
                        self._convert_schema_node(items, output, level + 3, "")
                    else:
                        # Array of primitives
                        item_type = (
                            items.get("type", "string") if isinstance(items, dict) else "string"
                        )
                        output.append(f"{indent}    - # Array of {item_type}")
                else:
                    # Primitive type
                    default_val = prop_schema.get("default", "")
                    if default_val:
                        output.append(f"{indent}  {prop_name}: {default_val}")
                    else:
                        output.append(f"{indent}  {prop_name}:")

        elif schema_type == "array":
            items = schema.get("items", {})
            output.append(f"{indent}  # Array of:")
            if isinstance(items, dict):
                self._convert_schema_node(items, output, level, "")

    def yaml_to_json(self, json_str: str, schema: dict) -> dict:
        """Convert JSON string (containing YAML-formatted extraction) to typed JSON.

        Args:
            json_str: JSON string containing extracted data from LLM
            schema: JSON Schema to apply type constraints

        Returns:
            Dictionary with properly typed values
        """
        import json as json_module

        try:
            data = json_module.loads(json_str)
        except (json_module.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
            return {}

        if not isinstance(data, dict):
            return data

        return self._apply_schema_types(data, schema)

    def _apply_schema_types(self, data: dict, schema: dict) -> dict:
        """Convert data types to match the schema."""
        if not isinstance(data, dict):
            return data

        result = {}
        properties = schema.get("properties", {})

        for key, value in data.items():
            if key not in properties:
                result[key] = value
                continue

            prop_schema = properties[key]
            prop_type = prop_schema.get("type")

            if value is None:
                result[key] = None
            elif prop_type == "string":
                result[key] = str(value)
            elif prop_type == "integer":
                if value is None or value == "":
                    result[key] = None
                else:
                    try:
                        result[key] = int(float(value))
                    except (ValueError, TypeError):
                        result[key] = None
            elif prop_type == "number":
                if value is None or value == "":
                    result[key] = None
                else:
                    try:
                        result[key] = float(value)
                    except (ValueError, TypeError):
                        result[key] = None
            elif prop_type == "boolean":
                if isinstance(value, str):
                    result[key] = value.lower() in ("true", "yes", "1")
                else:
                    result[key] = bool(value)
            elif prop_type == "array":
                if isinstance(value, list):
                    items_schema = prop_schema.get("items", {})
                    result[key] = [
                        self._apply_schema_types(v, items_schema) if isinstance(v, dict) else v
                        for v in value
                    ]
                else:
                    result[key] = [value]
            elif prop_type == "object":
                if isinstance(value, dict):
                    result[key] = self._apply_schema_types(value, prop_schema)
                else:
                    result[key] = value
            else:
                result[key] = value

        return result
