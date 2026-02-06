"""Auto-discovery and introspection of YAML custom tags.

This module provides functions to discover available YAML custom tags
and extract documentation from their constructors.

Usage
-----
>>> from hexdag.core.pipeline_builder.tag_discovery import discover_tags
>>> tags = discover_tags()
>>> tags["!py"]["name"]
'!py'
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Any

import yaml

# Tag metadata registry
# Maps tag name -> (module, constructor_func, description)
_TAG_REGISTRY: dict[str, tuple[str, str, str]] = {
    "!py": (
        "hexdag.core.pipeline_builder.py_tag",
        "py_constructor",
        "Compile inline Python code into callable functions",
    ),
    "!include": (
        "hexdag.core.pipeline_builder.include_tag",
        "include_constructor",
        "Include content from external YAML files",
    ),
}


@lru_cache(maxsize=1)
def discover_tags() -> dict[str, dict[str, Any]]:
    """Discover all registered YAML custom tags with their metadata.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of tag name to tag info dict with keys:
        - name: Tag name (e.g., "!py")
        - module: Full module path
        - constructor: Constructor function name
        - description: Short description
        - documentation: Full docstring
        - syntax: Syntax patterns
        - is_registered: Whether tag is registered with YAML SafeLoader

    Examples
    --------
    >>> tags = discover_tags()
    >>> "!py" in tags
    True
    >>> tags["!py"]["description"]
    'Compile inline Python code into callable functions'
    """
    result: dict[str, dict[str, Any]] = {}

    for tag_name, (module_path, constructor_name, description) in _TAG_REGISTRY.items():
        tag_info = _extract_tag_info(tag_name, module_path, constructor_name, description)
        result[tag_name] = tag_info

    return result


def _extract_tag_info(
    tag_name: str,
    module_path: str,
    constructor_name: str,
    description: str,
) -> dict[str, Any]:
    """Extract comprehensive information about a tag.

    Parameters
    ----------
    tag_name : str
        The YAML tag (e.g., "!py")
    module_path : str
        Full module path to the tag module
    constructor_name : str
        Name of the constructor function
    description : str
        Short description

    Returns
    -------
    dict[str, Any]
        Tag information dictionary
    """
    tag_info: dict[str, Any] = {
        "name": tag_name,
        "module": module_path,
        "constructor": constructor_name,
        "description": description,
        "documentation": "",
        "syntax": [],
        "is_registered": tag_name in yaml.SafeLoader.yaml_constructors,
    }

    try:
        module = importlib.import_module(module_path)
        constructor = getattr(module, constructor_name, None)

        if constructor:
            # Extract full docstring from constructor
            docstring = inspect.getdoc(constructor) or ""
            tag_info["documentation"] = docstring

        # Also get module-level docstring for additional context
        module_doc = inspect.getdoc(module) or ""
        if module_doc and not tag_info["documentation"]:
            tag_info["documentation"] = module_doc

        # Extract syntax patterns
        tag_info["syntax"] = _get_tag_syntax(tag_name)

    except (ImportError, AttributeError) as e:
        tag_info["error"] = str(e)

    return tag_info


def _get_tag_syntax(tag_name: str) -> list[str]:
    """Get syntax patterns for a tag.

    Parameters
    ----------
    tag_name : str
        The tag name (e.g., "!py")

    Returns
    -------
    list[str]
        List of syntax pattern descriptions
    """
    if tag_name == "!py":
        return [
            "!py | <python_code>  # Inline Python code block",
            "The code block must define exactly one function",
            "Function signature: async def process(item, index, state, **ports)",
        ]
    if tag_name == "!include":
        return [
            "!include ./path/to/file.yaml  # Simple file inclusion",
            "!include {path: ./file.yaml, vars: {key: value}}  # With variable substitution",
            "Variables use {{var}} placeholder syntax in included files",
        ]
    return []


def get_tag_schema(tag_name: str) -> dict[str, Any]:
    """Get JSON Schema-like representation of a tag's usage.

    Parameters
    ----------
    tag_name : str
        Tag name (e.g., "!py" or "!include")

    Returns
    -------
    dict[str, Any]
        Schema-like dict with tag information

    Raises
    ------
    ValueError
        If tag is not found

    Examples
    --------
    >>> schema = get_tag_schema("!py")
    >>> schema["name"]
    '!py'
    >>> schema["type"]
    'yaml_tag'
    """
    # Normalize tag name (add ! prefix if missing)
    normalized_name = tag_name if tag_name.startswith("!") else f"!{tag_name}"

    tags = discover_tags()
    if normalized_name not in tags:
        available = list(tags.keys())
        raise ValueError(f"Unknown tag: {tag_name}. Available: {available}")

    tag_info = tags[normalized_name]

    # Build a schema-like representation
    schema: dict[str, Any] = {
        "name": tag_info["name"],
        "type": "yaml_tag",
        "description": tag_info["description"],
        "module": tag_info["module"],
        "documentation": tag_info["documentation"],
        "syntax": tag_info["syntax"],
        "is_registered": tag_info["is_registered"],
    }

    # Add tag-specific schema information
    if normalized_name == "!py":
        schema["input_schema"] = {
            "type": "string",
            "format": "python_code",
            "description": "Python source code defining a single function",
            "examples": ["async def process(item, index, state, **ports):\n    return item * 2"],
        }
        schema["output"] = {
            "type": "callable",
            "description": "Compiled Python function",
        }
        schema["security_warning"] = (
            "Executes arbitrary Python code. Only use with trusted YAML files."
        )

    elif normalized_name == "!include":
        schema["input_schema"] = {
            "oneOf": [
                {
                    "type": "string",
                    "description": "Path to YAML file to include",
                    "examples": ["./shared/nodes.yaml"],
                },
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to YAML file",
                        },
                        "vars": {
                            "type": "object",
                            "description": "Variables to substitute using {{var}} syntax",
                        },
                    },
                    "required": ["path"],
                },
            ],
        }
        schema["output"] = {
            "type": "any",
            "description": "Parsed YAML content from the included file",
        }

    return schema


def get_known_tag_names() -> frozenset[str]:
    """Get all registered tag names for validation.

    Returns
    -------
    frozenset[str]
        Set of all tag names (e.g., {"!py", "!include"})

    Examples
    --------
    >>> names = get_known_tag_names()
    >>> "!py" in names
    True
    >>> "!include" in names
    True
    """
    return frozenset(discover_tags().keys())
