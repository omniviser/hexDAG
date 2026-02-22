"""!include YAML custom tag handler for pipeline composition.

This module provides a YAML custom tag constructor that includes content
from external YAML files, enabling modular pipeline composition.

Examples
--------
Include a list of nodes::

    spec:
      nodes:
        - kind: expression_node
          metadata:
            name: start
          spec:
            expressions:
              ready: "true"

        # Include nodes from external file
        - !include ./shared/validation_nodes.yaml

        - kind: llm_node
          metadata:
            name: final
          spec:
            prompt_template: "Finalize: {{input}}"

Include a partial pipeline fragment::

    # In shared/validation_nodes.yaml
    - kind: function_node
      metadata:
        name: validate_input
      spec:
        fn: "myapp.validate"

    - kind: expression_node
      metadata:
        name: check_result
      spec:
        expressions:
          valid: "validate_input.success"

Include with variable substitution::

    # Use !include with a mapping for variable substitution
    - !include
      path: ./templates/processor.yaml
      vars:
        node_name: "custom_processor"
        timeout: 30
"""

from pathlib import Path
from typing import Any

import yaml

from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Thread-local storage for base path during parsing
_current_base_path: Path | None = None


class IncludeTagError(HexDAGError):
    """Error including external YAML file."""

    pass


def set_include_base_path(base_path: Path | None) -> None:
    """Set the base path for resolving !include paths.

    This should be called before parsing YAML that may contain !include tags.

    Parameters
    ----------
    base_path : Path | None
        Base directory for resolving relative paths
    """
    global _current_base_path
    _current_base_path = base_path
    if base_path:
        logger.debug("Set include base path", base_path=str(base_path))


def get_include_base_path() -> Path:
    """Get the current base path for !include resolution.

    Returns
    -------
    Path
        Current base path (defaults to cwd if not set)
    """
    return _current_base_path or Path.cwd()


def include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Include content from an external YAML file.

    Supports two forms:
    1. Simple: !include ./path/to/file.yaml
    2. With vars: !include {path: ./file.yaml, vars: {key: value}}

    Parameters
    ----------
    loader : yaml.SafeLoader
        YAML loader instance
    node : yaml.Node
        YAML node (scalar for simple, mapping for vars)

    Returns
    -------
    Any
        Parsed content from the included file

    Raises
    ------
    IncludeTagError
        If the file cannot be found or parsed
    """
    base_path = get_include_base_path()

    # Handle simple scalar form: !include ./path.yaml
    if isinstance(node, yaml.ScalarNode):
        include_path = loader.construct_scalar(node)
        if not isinstance(include_path, str):
            raise IncludeTagError("!include path must be a string")
        vars_dict: dict[str, Any] = {}

    # Handle mapping form: !include {path: ./path.yaml, vars: {...}}
    elif isinstance(node, yaml.MappingNode):
        # Use deep=True to fully construct nested mappings (like vars)
        mapping = loader.construct_mapping(node, deep=True)
        path_value = mapping.get("path")
        if not path_value:
            raise IncludeTagError("!include mapping requires 'path' key")
        include_path = str(path_value)
        vars_dict = mapping.get("vars", {})

    else:
        raise IncludeTagError(f"!include expects scalar or mapping, got {type(node)}")

    # Resolve path relative to base
    resolved_path = _resolve_include_path(include_path, base_path)

    logger.debug(
        "Including file",
        include_path=include_path,
        resolved_path=str(resolved_path),
        has_vars=bool(vars_dict),
    )

    # Read and parse the file
    try:
        content = resolved_path.read_text()
    except FileNotFoundError as e:
        raise IncludeTagError(
            f"Include file not found: {resolved_path}\n"
            f"  (resolved from '{include_path}' relative to '{base_path}')"
        ) from e
    except Exception as e:
        raise IncludeTagError(f"Failed to read include file {resolved_path}: {e}") from e

    # Apply variable substitution if vars provided
    if vars_dict:
        content = _substitute_vars(content, vars_dict)

    # Parse the included content
    try:
        # Set base path for nested includes
        old_base = _current_base_path
        set_include_base_path(resolved_path.parent)
        try:
            result = yaml.safe_load(content)
        finally:
            set_include_base_path(old_base)
    except yaml.YAMLError as e:
        raise IncludeTagError(f"Failed to parse include file {resolved_path}: {e}") from e

    return result


def _resolve_include_path(include_path: str, base_path: Path) -> Path:
    """Resolve include path relative to base path.

    Parameters
    ----------
    include_path : str
        Path from !include tag (may be relative or absolute)
    base_path : Path
        Base directory for relative paths

    Returns
    -------
    Path
        Resolved absolute path
    """
    path = Path(include_path)

    # If absolute, use as-is
    if path.is_absolute():
        return path

    # Resolve relative to base path
    return (base_path / path).resolve()


def _substitute_vars(content: str, vars_dict: dict[str, Any]) -> str:
    """Substitute variables in content using {{var}} syntax.

    Parameters
    ----------
    content : str
        YAML content with {{var}} placeholders
    vars_dict : dict[str, Any]
        Variable values to substitute

    Returns
    -------
    str
        Content with variables substituted
    """
    result = content
    for key, value in vars_dict.items():
        placeholder = "{{" + key + "}}"
        # Convert value to YAML-safe string representation
        if isinstance(value, str):
            str_value = value
        elif isinstance(value, bool):
            str_value = "true" if value else "false"
        elif value is None:
            str_value = "null"
        else:
            str_value = str(value)
        result = result.replace(placeholder, str_value)
    return result


def register_include_tag() -> None:
    """Register the !include custom tag with YAML SafeLoader.

    This function should be called during module initialization to enable
    !include tag support in YAML parsing.

    Examples
    --------
    Import the module to auto-register::

        import hexdag.kernel.pipeline_builder.include_tag  # Registers !include tag

    Or explicitly register::

        from hexdag.kernel.pipeline_builder.include_tag import register_include_tag
        register_include_tag()
    """
    # Check if already registered to avoid duplicate registration
    if "!include" not in yaml.SafeLoader.yaml_constructors:
        yaml.SafeLoader.add_constructor("!include", include_constructor)
        logger.debug("Registered !include YAML tag")


# Auto-register when module is imported
register_include_tag()
