"""Schema Converter - JSON Schema to YAML transformation.

Converts JSON Schema to YAML format optimized for LLM comprehension.
Uses pyyaml for all conversions - no LLM calls.
"""

import logging

logger = logging.getLogger(__name__)


def json_to_yaml(json_schema: dict, compact: bool = False) -> str:
    """Convert JSON Schema to YAML format optimized for LM comprehension.

    Args:
        json_schema: JSON Schema as a dict
        compact: If True, use compact format with minimal descriptions to save tokens

    Returns:
        YAML string with type hints, descriptions, and clear structure
    """
    yaml_lines = ["# Extraction Schema", "# Extract the following fields from the document:\n"]
    _convert_schema_node(json_schema, yaml_lines, level=0, compact=compact)
    return "\n".join(yaml_lines)


def _convert_schema_node(schema: dict, output: list[str], level: int = 0, name: str = "", compact: bool = False) -> None:
    """Recursively convert a schema node to YAML representation."""
    indent = "  " * level

    if name:
        output.append(f"{indent}{name}:")

    schema_type = schema.get("type")
    description = schema.get("description", "")
    title = schema.get("title", "")

    # Add metadata as comments (skip in compact mode for nested fields)
    if not compact or level == 0:
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

            prop_type = prop_schema.get("type", "string")
            prop_desc = prop_schema.get("description", "")
            prop_pattern = prop_schema.get("pattern", "")
            prop_format = prop_schema.get("format", "")
            prop_enum = prop_schema.get("enum", [])

            type_comment = prop_type
            if prop_format:
                type_comment += f" (format: {prop_format})"
            if prop_pattern:
                type_comment += f" (pattern: {prop_pattern})"
            if prop_enum:
                type_comment += f" (enum: {', '.join(map(str, prop_enum))})"

            # In compact mode, just show type and required status
            if compact and level > 0:
                output.append(f"{indent}  # {prop_name}: {type_comment}{req_marker}")
            else:
                output.append(f"{indent}  # {prop_name}: {type_comment}{req_marker}")
                if prop_desc:
                    output.append(f"{indent}  #   {prop_desc}")

            if prop_type == "object" and "properties" in prop_schema:
                output.append(f"{indent}  {prop_name}:")
                _convert_schema_node(prop_schema, output, level + 2, "", compact=compact)
            elif prop_type == "array":
                items = prop_schema.get("items", {})
                output.append(f"{indent}  {prop_name}:")
                if isinstance(items, dict) and "properties" in items:
                    output.append(f"{indent}    - # Array of objects with:")
                    _convert_schema_node(items, output, level + 3, "", compact=compact)
                else:
                    item_type = items.get("type", "string") if isinstance(items, dict) else "string"
                    output.append(f"{indent}    - # Array of {item_type}")
            else:
                default_val = prop_schema.get("default", "")
                if default_val:
                    output.append(f"{indent}  {prop_name}: {default_val}")
                else:
                    output.append(f"{indent}  {prop_name}:")

    elif schema_type == "array":
        items = schema.get("items", {})
        output.append(f"{indent}  # Array of:")
        if isinstance(items, dict):
            _convert_schema_node(items, output, level, "", compact=compact)


def yaml_to_json(json_str: str, schema: dict) -> dict:
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

    return _apply_schema_types(data, schema)


def _to_int(value):
    """Safely convert value to int."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_float(value):
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _to_bool(value):
    """Safely convert value to bool."""
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


# Type handler dispatch table
_TYPE_HANDLERS = {
    "string": lambda v: str(v) if v else None,
    "integer": lambda v: None if v in (None, "") else _to_int(v),
    "number": lambda v: None if v in (None, "") else _to_float(v),
    "boolean": _to_bool,
}


def _apply_schema_types(data: dict, schema: dict) -> dict:
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
        elif prop_type in _TYPE_HANDLERS:
            result[key] = _TYPE_HANDLERS[prop_type](value)
        elif prop_type == "array":
            items_schema = prop_schema.get("items", {})
            if isinstance(value, list):
                result[key] = [
                    _apply_schema_types(v, items_schema) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = [value]
        elif prop_type == "object":
            if isinstance(value, dict):
                result[key] = _apply_schema_types(value, prop_schema)
            else:
                result[key] = value
        else:
            result[key] = value

    return result
