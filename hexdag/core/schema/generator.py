"""Schema generator - converts Python signatures to JSON Schema."""

import contextlib
import inspect
import json
import re
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from hexdag.core.logging import get_logger
from hexdag.core.types import (
    get_annotated_metadata,
    is_annotated_type,
    is_dict_type,
    is_list_type,
    is_literal_type,
    is_union_type,
)

logger = get_logger(__name__)


class SchemaGenerator:
    """Generate JSON Schema from Python callables.

    This class introspects Python functions/methods to automatically generate
    JSON Schema definitions. It supports:
    - Basic types (str, int, float, bool)
    - Literal types → enum
    - Union types → anyOf
    - List/Dict types → array/object
    - Annotated types with Pydantic Field constraints
    - Docstring extraction for descriptions

    Examples
    --------
    >>> def my_func(name: str, count: int = 10):
    ...     '''Example function.'''
    ...     pass
    >>> schema = SchemaGenerator.from_callable(my_func)
    >>> schema['properties']['count']['default']
    10
    """

    # Basic type mapping from Python to JSON Schema
    BASIC_TYPE_MAP = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
        None: {"type": "null"},
        type(None): {"type": "null"},
    }

    # Framework parameters to skip when generating config schemas for nodes.
    # These are internal/structural parameters, not user-configurable options.
    FRAMEWORK_PARAMS = {
        # Node structure params
        "name",
        "deps",
        "dependencies",
        # Function node internal params (fn defines behavior, schemas are inferred)
        "fn",
        "input_schema",
        "input_mapping",
        "unpack_input",
    }

    @staticmethod
    def from_callable(factory: Callable, format: str = "dict") -> dict | str:
        """Generate schema from factory __call__ signature.

        Resolution order:
        1. Check for explicit ``_yaml_schema`` class attribute
        2. Fall back to ``__call__`` signature introspection

        This allows builder-pattern classes like ConditionalNode to define
        explicit schemas for YAML/MCP usage.

        Args
        ----
            factory: Callable (function, method, or callable class) to introspect
            format: Output format - "dict", "yaml", or "json"

        Returns
        -------
            dict | str: JSON Schema in requested format

        Raises
        ------
        ValueError
            If format is not one of: dict, yaml, json

        Examples
        --------
        >>> def factory(name: str, count: int, enabled: bool = True):
        ...     pass
        >>> schema = SchemaGenerator.from_callable(factory)
        >>> schema['properties']['enabled']['default']
        True
        >>> schema['required']
        ['count']

        >>> # Classes can define explicit schemas
        >>> class MyNode:
        ...     _yaml_schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        >>> schema = SchemaGenerator.from_callable(MyNode)
        >>> "foo" in schema.get("properties", {})
        True
        """
        if format not in ("dict", "yaml", "json"):
            raise ValueError(f"Invalid format: {format}. Must be one of: dict, yaml, json")

        # Check for explicit _yaml_schema class attribute (for builder-pattern nodes)
        yaml_schema = getattr(factory, "_yaml_schema", None)
        if yaml_schema and isinstance(yaml_schema, dict):
            logger.debug(f"Using explicit _yaml_schema for {factory}")
            return SchemaGenerator._format_output(yaml_schema, format)

        try:
            sig = inspect.signature(factory)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not get signature for {factory}: {e}")
            return SchemaGenerator._format_output({}, format)

        properties = {}
        required = []

        # Extract param docs - try __call__ method first (for callable instances),
        # then fall back to the factory itself
        param_docs: dict[str, str] = {}
        # We need the actual __call__ method to extract docstrings, not just callable check
        call_method = getattr(factory, "__call__", None)  # noqa: B004
        if call_method is not None:
            param_docs = SchemaGenerator._extract_param_docs(call_method)
        if not param_docs:
            param_docs = SchemaGenerator._extract_param_docs(factory)

        param_list = list(sig.parameters.items())
        first_non_self_param = None
        for pname, _ in param_list:
            if pname not in ("self", "cls"):
                first_non_self_param = pname
                break

        # Resolve string annotations to actual types (handles PEP 563)
        # Use include_extras=True to preserve Annotated metadata for Field constraints
        type_hints: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            type_hints = get_type_hints(factory, include_extras=True)

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name in ("self", "cls", "args", "kwargs"):
                continue

            # Skip 'name' if it's the first parameter (node factory convention)
            if param_name == "name" and param_name == first_non_self_param:
                continue

            # Skip framework/structural parameters (not user config)
            if param_name in SchemaGenerator.FRAMEWORK_PARAMS:
                continue

            # Skip *args and **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # Get resolved type from type_hints, fall back to annotation
            param_type = type_hints.get(param_name, param.annotation)

            # Skip if no type annotation
            if param_type == inspect.Parameter.empty:
                # Default to string type
                param_type = str

            # Generate property schema
            prop_schema = SchemaGenerator._type_to_json_schema(param_type)

            if param_name in param_docs:
                doc_text = param_docs[param_name]
                prop_schema["description"] = doc_text

                # For list[dict] types, try to extract nested structure from docstring
                if (
                    prop_schema.get("type") == "array"
                    and prop_schema.get("items", {}).get("type") == "object"
                ):
                    nested_schema = SchemaGenerator._extract_nested_structure(doc_text)
                    if nested_schema and "properties" in nested_schema:
                        prop_schema["items"] = nested_schema

            if param.default != inspect.Parameter.empty:
                prop_schema["default"] = param.default
            else:
                # Required if no default
                required.append(param_name)

            properties[param_name] = prop_schema

        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

        if required:
            schema["required"] = required

        return SchemaGenerator._format_output(schema, format)

    @staticmethod
    def _is_callable_type(type_hint: Any) -> bool:
        """Check if type hint is a Callable type.

        Examples
        --------
        >>> from collections.abc import Callable
        >>> SchemaGenerator._is_callable_type(Callable)
        True
        >>> SchemaGenerator._is_callable_type(Callable[..., Any])
        True
        >>> SchemaGenerator._is_callable_type(str)
        False
        """
        # Check for collections.abc.Callable
        if type_hint is Callable:
            return True

        # Check for typing.Callable or parameterized Callable[..., ...]
        origin = get_origin(type_hint)
        if origin is Callable:
            return True

        # Check for callable origin from collections.abc
        try:
            from collections.abc import Callable as ABCCallable

            if origin is ABCCallable:
                return True
        except ImportError:
            pass

        return False

    @staticmethod
    def _type_to_json_schema(type_hint: Any) -> dict:
        """Convert Python type hint to JSON Schema type.

        Handles:
        - Basic types: str, int, float, bool
        - Literal types: Literal["a", "b"] → enum
        - Union types: str | int → anyOf
        - List types: list[str] → array
        - Dict types: dict[str, Any] → object
        - Callable types: Callable[..., Any] → string (module path)
        - Annotated types: Annotated[int, Field(ge=0)] → min/max
        - Type aliases: Resolves type aliases to their base Literal/Union types

        Args
        ----
            type_hint: Python type annotation

        Returns
        -------
            dict: JSON Schema definition for the type

        Examples
        --------
        >>> from typing import Literal
        >>> SchemaGenerator._type_to_json_schema(Literal["a", "b"])
        {'type': 'string', 'enum': ['a', 'b']}
        """
        # Handle Callable types first (before checking other patterns)
        # Callables are represented as strings (module paths) in YAML
        if SchemaGenerator._is_callable_type(type_hint):
            return {
                "type": "string",
                "description": "Module path string (e.g., 'myapp.process') or !py inline function",
            }

        if is_annotated_type(type_hint):
            base_type, metadata = get_annotated_metadata(type_hint)

            # Recursively process base type
            schema = SchemaGenerator._type_to_json_schema(base_type)

            # In Pydantic v2, Field stores metadata as FieldInfo with .metadata attribute
            for constraint in metadata:
                if hasattr(constraint, "metadata") and constraint.metadata:
                    for meta_item in constraint.metadata:
                        # Ge, Le, Gt, Lt objects
                        if hasattr(meta_item, "ge"):
                            schema["minimum"] = meta_item.ge
                        if hasattr(meta_item, "le"):
                            schema["maximum"] = meta_item.le
                        if hasattr(meta_item, "gt"):
                            schema["exclusiveMinimum"] = meta_item.gt
                        if hasattr(meta_item, "lt"):
                            schema["exclusiveMaximum"] = meta_item.lt
                        # MinLen, MaxLen objects
                        if hasattr(meta_item, "min_length"):
                            schema["minLength"] = meta_item.min_length
                        if hasattr(meta_item, "max_length"):
                            schema["maxLength"] = meta_item.max_length

                # Also check for description on the Field itself
                if hasattr(constraint, "description") and constraint.description:
                    schema["description"] = constraint.description

            return schema

        if is_literal_type(type_hint):
            args = get_args(type_hint)
            # Determine type from first value
            first_val = args[0] if args else ""
            val_type = type(first_val)

            # Determine JSON Schema type from Python type
            if val_type in SchemaGenerator.BASIC_TYPE_MAP:
                json_type = SchemaGenerator.BASIC_TYPE_MAP[val_type]["type"]
            else:
                json_type = "string"

            literal_schema: dict[str, Any] = {"type": json_type, "enum": list(args)}

            return literal_schema

        if is_union_type(type_hint):
            args = get_args(type_hint)

            # Filter out None for Optional types
            non_none_args = [arg for arg in args if arg is not type(None)]

            if len(non_none_args) == 1:
                # Optional[T] → make nullable
                schema = SchemaGenerator._type_to_json_schema(non_none_args[0])
                # Allow null
                if "type" in schema:
                    if isinstance(schema["type"], list):
                        schema["type"].append("null")
                    else:
                        schema["type"] = [schema["type"], "null"]
                return schema

            # Multiple types → anyOf
            # But first, check if any of them are Callable and simplify
            processed_schemas = [SchemaGenerator._type_to_json_schema(arg) for arg in non_none_args]

            # Deduplicate schemas that have the same structure
            unique_schemas = []
            seen = set()
            for schema in processed_schemas:
                # Create a hashable representation
                schema_repr = json.dumps(schema, sort_keys=True)
                if schema_repr not in seen:
                    seen.add(schema_repr)
                    unique_schemas.append(schema)

            if len(unique_schemas) == 1:
                return unique_schemas[0]

            return {"anyOf": unique_schemas}

        if is_list_type(type_hint):
            args = get_args(type_hint)
            item_type = args[0] if args else Any

            return {
                "type": "array",
                "items": SchemaGenerator._type_to_json_schema(item_type),
            }

        if is_dict_type(type_hint):
            args = get_args(type_hint)
            # If dict has typed values, try to extract schema
            if len(args) >= 2:
                value_type = args[1]
                if value_type is not Any:
                    return {
                        "type": "object",
                        "additionalProperties": SchemaGenerator._type_to_json_schema(value_type),
                    }
            return {"type": "object"}

        if type_hint in SchemaGenerator.BASIC_TYPE_MAP:
            return SchemaGenerator.BASIC_TYPE_MAP[type_hint].copy()

        # Check if it's a type alias that we can resolve
        # Type aliases like `Mode = Literal[...]` should be resolved
        if hasattr(type_hint, "__value__"):
            # Python 3.12+ type aliases have __value__
            return SchemaGenerator._type_to_json_schema(type_hint.__value__)

        # Default to string for unknown types
        return {"type": "string"}

    @staticmethod
    def _extract_nested_structure(description: str) -> dict[str, Any] | None:
        """Extract nested object structure from parameter description.

        Parses docstring descriptions that define object structures with fields:
        - "Each branch has: - condition: str - Expression..."
        - "Dict with keys: field1: type - description..."

        Args
        ----
            description: Parameter description text

        Returns
        -------
            dict[str, Any] | None: JSON Schema properties dict if structure found, None otherwise

        Examples
        --------
        >>> desc = "List of branches. Each has: - condition: str - The condition"
        >>> result = SchemaGenerator._extract_nested_structure(desc)
        >>> "condition" in result.get("properties", {}) if result else False
        True
        """
        if not description:
            return None

        properties: dict[str, Any] = {}

        # Pattern 1: "- field: type - description" (bullet list style)
        # Matches lines like "- condition: str - Expression to evaluate"
        bullet_pattern = re.compile(
            r"-\s+(\w+):\s*(\w+)?\s*[-–—]?\s*(.*?)(?=\n\s*-|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        matches = bullet_pattern.findall(description)

        for match in matches:
            field_name = match[0].strip()
            field_type = match[1].strip() if match[1] else "string"
            field_desc = match[2].strip() if len(match) > 2 else ""

            # Map common type names to JSON Schema types
            type_map = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }

            json_type = type_map.get(field_type.lower(), "string")
            prop_schema: dict[str, Any] = {"type": json_type}
            if field_desc:
                prop_schema["description"] = field_desc

            properties[field_name] = prop_schema

        if properties:
            return {"type": "object", "properties": properties}

        return None

    @staticmethod
    def _extract_param_docs(func: Callable) -> dict[str, str]:
        """Extract parameter descriptions from function docstring.

        Supports multiple docstring formats:
        - Google style (Args:)
        - NumPy style (Parameters with --- separator)
        - Sphinx style (:param name:)

        Args
        ----
            func: Function to extract docs from

        Returns
        -------
            dict[str, str]: Mapping of parameter name to description

        Examples
        --------
        >>> def func(name: str, count: int):
        ...     '''Function description.
        ...
        ...     Args:
        ...         name: The name parameter
        ...         count: The count parameter
        ...     '''
        ...     pass
        >>> docs = SchemaGenerator._extract_param_docs(func)
        >>> docs['name']
        'The name parameter'
        """
        docstring = inspect.getdoc(func)
        if not docstring:
            return {}

        param_docs: dict[str, str] = {}
        lines = docstring.split("\n")

        # Look for Args/Parameters section
        in_params_section = False
        current_param: str | None = None
        is_numpy_style = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check for NumPy-style separator (line of dashes after Parameters header)
            # Also check if this is a short separator (4 chars like "----") which may be
            # a formatting artifact rather than a true NumPy-style separator
            if (
                in_params_section
                and line_stripped
                and len(line_stripped) >= 3
                and all(c in ("-", "=", "_") for c in line_stripped)
            ):
                # Only treat as NumPy style if the separator is long enough (>=10 chars)
                # Short separators like "----" are often just formatting
                if len(line_stripped) >= 10:
                    is_numpy_style = True
                continue

            # Detect start of parameters section (case insensitive)
            if line_stripped.lower() in (
                "args:",
                "arguments:",
                "parameters:",
                "params:",
                "args",
                "arguments",
                "parameters",
                "params",
            ):
                in_params_section = True
                continue

            # Exit parameters section when we hit another section header
            if in_params_section:
                # NumPy style: section headers are followed by separator lines
                # Check if next line is a separator (indicating new section)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if (
                        line_stripped
                        and not line.startswith((" ", "\t"))
                        and next_line
                        and len(next_line) >= 3
                        and all(c in ("-", "=", "_") for c in next_line)
                    ):
                        break

                # Google style: section headers end with :
                if (
                    line_stripped
                    and line_stripped.endswith(":")
                    and not line.startswith(" ")
                    and line_stripped.lower() not in ("args:", "parameters:", "params:")
                ):
                    break

                # Check for section keywords
                if (
                    line
                    and not line.startswith((" ", "\t"))
                    and line_stripped
                    and any(
                        keyword in line_stripped.lower()
                        for keyword in [
                            "returns",
                            "raises",
                            "yields",
                            "examples",
                            "notes",
                            "see also",
                        ]
                    )
                ):
                    break

            # Process parameter lines
            if in_params_section:
                # Sphinx style: ":param param_name: description"
                if line_stripped.startswith(":param"):
                    parts = line_stripped.split(":", 3)
                    if len(parts) >= 3:
                        param_name = parts[1].replace("param", "").strip()
                        description = parts[2].strip()
                        param_docs[param_name] = description
                        current_param = param_name
                # NumPy style: "param_name : type" on one line, description indented below
                elif is_numpy_style and " : " in line_stripped and not line.startswith((" ", "\t")):
                    parts = line_stripped.split(" : ", 1)
                    param_name = parts[0].strip()
                    # Skip type info, description comes on next indented lines
                    if param_name and not any(
                        keyword in param_name.lower()
                        for keyword in ["return", "raise", "yield", "example", "note"]
                    ):
                        param_docs[param_name] = ""
                        current_param = param_name
                # NumPy style continuation: indented lines are descriptions
                elif (
                    is_numpy_style
                    and current_param
                    and line.startswith((" ", "\t"))
                    and line_stripped
                ):
                    if param_docs[current_param]:
                        param_docs[current_param] += " " + line_stripped
                    else:
                        param_docs[current_param] = line_stripped
                # Google style: "param_name: description" or "param_name (type): description"
                # Only applies when NOT in NumPy mode (to avoid conflicts)
                elif not is_numpy_style and ":" in line_stripped:
                    parts = line_stripped.split(":", 1)
                    if len(parts) == 2:
                        param_part = parts[0].strip()
                        description = parts[1].strip()
                        # Handle "param_name (type)" format
                        if "(" in param_part:
                            param_name = param_part.split("(")[0].strip()
                        else:
                            param_name = param_part
                        # Make sure it's not a section header
                        if param_name and not any(
                            keyword in param_name.lower()
                            for keyword in ["return", "raise", "yield", "example", "note"]
                        ):
                            param_docs[param_name] = description
                            current_param = param_name
                # Google style continuation lines (indented description for current param)
                elif (
                    not is_numpy_style
                    and current_param
                    and line.startswith((" ", "\t"))
                    and line_stripped
                ):
                    if param_docs[current_param]:
                        param_docs[current_param] += " " + line_stripped
                    else:
                        param_docs[current_param] = line_stripped

        return param_docs

    @staticmethod
    def _format_output(schema: dict, format: str) -> dict | str:
        """Format schema output as dict, YAML, or JSON.

        Args
        ----
            schema: JSON Schema dict
            format: Output format - "dict", "yaml", or "json"

        Returns
        -------
            dict | str: Schema in requested format
        """
        if format == "dict":
            return schema
        if format == "yaml":
            yaml_str: str = yaml.dump(schema, sort_keys=False, default_flow_style=False)
            return yaml_str
        if format == "json":
            return json.dumps(schema, indent=2)
        return schema

    @staticmethod
    def generate_example_yaml(node_type: str, schema: dict) -> str:
        """Generate example YAML from schema.

        Creates a complete YAML example with:
        - K8s-style structure (kind, metadata, spec)
        - Default values where available
        - Placeholders for required fields
        - Comments for optional fields

        Args
        ----
            node_type: Node type name (e.g., "llm_node")
            schema: JSON Schema dict

        Returns
        -------
            str: Example YAML string

        Examples
        --------
        >>> schema = {
        ...     "properties": {
        ...         "template": {"type": "string"},
        ...         "model": {"type": "string", "default": "gpt-4"}
        ...     },
        ...     "required": ["template"]
        ... }
        >>> example = SchemaGenerator.generate_example_yaml("llm_node", schema)
        >>> "kind: llm_node" in example
        True
        """
        example: dict[str, Any] = {
            "kind": node_type,
            "metadata": {"name": f"my_{node_type}"},
            "spec": {},
        }

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        spec: dict[str, Any] = example["spec"]
        for prop_name, prop_schema in properties.items():
            if "default" in prop_schema:
                # Use default value
                spec[prop_name] = prop_schema["default"]
            elif "examples" in prop_schema:
                # Use first example
                spec[prop_name] = prop_schema["examples"][0]
            elif prop_name in required:
                # Required field - add placeholder
                spec[prop_name] = SchemaGenerator._placeholder_for_type(prop_schema.get("type"))
            # Optional fields without defaults are omitted

        yaml_output: str = yaml.dump(example, sort_keys=False, default_flow_style=False)
        return yaml_output

    @staticmethod
    def _placeholder_for_type(json_type: str | list | None) -> Any:
        """Get placeholder value for a JSON Schema type.

        Args
        ----
            json_type: JSON Schema type string or list

        Returns
        -------
            Any: Appropriate placeholder value

        Examples
        --------
        >>> SchemaGenerator._placeholder_for_type("string")
        'value'
        >>> SchemaGenerator._placeholder_for_type("integer")
        0
        """
        if isinstance(json_type, list):
            json_type = json_type[0] if json_type else "string"

        placeholders = {
            "string": "value",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
            "null": None,
        }

        return placeholders.get(json_type if isinstance(json_type, str) else "string", "value")
