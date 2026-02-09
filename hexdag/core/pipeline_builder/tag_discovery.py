"""Auto-discovery and introspection of YAML custom tags.

This module provides functions to discover available YAML custom tags
and extract documentation from their constructors.

Tags are discovered from:
1. Builtin tags in hexdag.core.pipeline_builder (modules ending with _tag)
2. Plugin tags registered via entry points (hexdag.tags group)

Plugin Tag Registration
-----------------------
Plugins can register custom YAML tags via pyproject.toml::

    [project.entry-points."hexdag.tags"]
    "!mytag" = "myplugin.tags:mytag_constructor"

The constructor function will receive (loader, node) arguments per PyYAML convention.

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
import pkgutil
from functools import lru_cache
from typing import Any

import yaml


def _discover_builtin_tags() -> dict[str, tuple[str, str, str]]:
    """Discover builtin tags from hexdag.core.pipeline_builder package.

    Scans for modules ending with '_tag' and looks for constructor functions
    following the naming pattern '<tagname>_constructor'.

    Returns
    -------
    dict[str, tuple[str, str, str]]
        Mapping of tag name to (module_path, constructor_name, description)
    """
    tags: dict[str, tuple[str, str, str]] = {}

    try:
        package = importlib.import_module("hexdag.core.pipeline_builder")
        if not hasattr(package, "__path__"):
            return tags

        for _finder, name, _ispkg in pkgutil.iter_modules(package.__path__):
            # Look for modules ending with _tag (e.g., py_tag, include_tag)
            if not name.endswith("_tag"):
                continue

            module_path = f"hexdag.core.pipeline_builder.{name}"

            try:
                module = importlib.import_module(module_path)
            except ImportError:
                continue

            # Derive tag name and constructor name from module name
            # py_tag -> !py, py_constructor
            # include_tag -> !include, include_constructor
            tag_base = name[:-4]  # Remove '_tag' suffix
            tag_name = f"!{tag_base}"
            constructor_name = f"{tag_base}_constructor"

            # Check if constructor exists in module
            constructor = getattr(module, constructor_name, None)
            if constructor is None:
                continue

            # Get description from constructor docstring
            description = (
                (inspect.getdoc(constructor) or f"YAML {tag_name} tag").split("\n")[0].strip()
            )

            tags[tag_name] = (module_path, constructor_name, description)

    except ImportError:
        pass

    return tags


def _discover_plugin_tags() -> dict[str, tuple[str, str, str]]:
    """Discover custom tags from plugin entry points.

    Plugins can register custom YAML tags via pyproject.toml::

        [project.entry-points."hexdag.tags"]
        "!mytag" = "myplugin.tags:mytag_constructor"

    Returns
    -------
    dict[str, tuple[str, str, str]]
        Mapping of tag name to (module_path, constructor_name, description)
    """
    tags: dict[str, tuple[str, str, str]] = {}

    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="hexdag.tags")
        for ep in eps:
            try:
                # Load the constructor to get its docstring
                constructor = ep.load()
                description = (
                    (inspect.getdoc(constructor) or "Custom tag from plugin").split("\n")[0].strip()
                )

                # Parse entry point value: "module.path:function_name"
                if ":" in ep.value:
                    module_path, func_name = ep.value.rsplit(":", 1)
                else:
                    # Fallback: assume the whole value is a module with __call__
                    module_path = ep.value
                    func_name = "__call__"

                # Normalize tag name (ensure ! prefix)
                tag_name = ep.name if ep.name.startswith("!") else f"!{ep.name}"
                tags[tag_name] = (module_path, func_name, description)
            except Exception:
                # Skip broken entry points silently
                pass
    except Exception:
        # importlib.metadata not available or entry_points failed
        pass

    return tags


@lru_cache(maxsize=1)
def discover_tags() -> dict[str, dict[str, Any]]:
    """Discover all registered YAML custom tags with their metadata.

    Discovers tags from:
    1. Builtin tags in hexdag.core.pipeline_builder (auto-discovered)
    2. Plugin tags via entry points (hexdag.tags group)

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
        - source: "builtin" or "plugin"

    Examples
    --------
    >>> tags = discover_tags()
    >>> "!py" in tags
    True
    >>> tags["!py"]["description"]
    'Compile !py tagged Python code into a callable function.'
    """
    result: dict[str, dict[str, Any]] = {}

    # Discover builtin tags dynamically
    for tag_name, (module_path, constructor_name, description) in _discover_builtin_tags().items():
        tag_info = _extract_tag_info(tag_name, module_path, constructor_name, description)
        tag_info["source"] = "builtin"
        result[tag_name] = tag_info

    # Discover plugin tags (can override builtins if desired)
    for tag_name, (module_path, constructor_name, description) in _discover_plugin_tags().items():
        tag_info = _extract_tag_info(tag_name, module_path, constructor_name, description)
        tag_info["source"] = "plugin"
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

        # Add security warning for tags that execute arbitrary code
        if tag_name == "!py":
            tag_info["security_warning"] = (
                "Executes arbitrary Python code. Only use with trusted YAML files."
            )

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
        # Add schema and yaml_example for MCP compatibility
        schema["schema"] = schema["input_schema"]
        schema["yaml_example"] = """body: !py |
  async def process(item, index, state, **ports):
      return item * 2"""

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
        # Add schema and yaml_example for MCP compatibility
        schema["schema"] = schema["input_schema"]
        schema["yaml_example"] = """# Simple include
nodes: !include ./shared/nodes.yaml

# Include with variable substitution
config: !include
  path: ./templates/config.yaml
  vars:
    env: production"""

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


def clear_tag_cache() -> None:
    """Clear the tag discovery cache.

    Useful for testing or when plugins are dynamically loaded/unloaded.
    """
    discover_tags.cache_clear()
