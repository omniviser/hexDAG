"""Pipeline validation API.

Provides unified functions for validating hexDAG YAML pipelines.
Delegates to core YamlValidator for structural validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from hexdag.compiler import YamlPipelineBuilder
from hexdag.compiler.yaml_validator import YamlValidator


def validate(
    yaml_content: str,
    lenient: bool = False,
    base_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a YAML pipeline configuration.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    lenient : bool
        If True, validate structure only without requiring environment variables.
        Useful for CI/CD validation where secrets aren't available.
        Default: False
    base_path : Path | None
        Base directory for resolving ``!include`` directives.
        Required for full validation of pipelines that use includes.
        Default: None (uses current working directory)

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - Whether the pipeline is valid
        - message: str - Success message if valid
        - error: str - Error message if invalid
        - error_type: str - Exception class name if invalid
        - node_count: int - Number of nodes (if valid)
        - nodes: list[str] - Node names (if valid)
        - ports: list[str] - Port names (if valid, full validation only)
        - warnings: list[str] - Validation warnings (lenient mode only)

    Examples
    --------
    >>> result = validate('''
    ... apiVersion: hexdag/v1
    ... kind: Pipeline
    ... metadata:
    ...   name: test
    ... spec:
    ...   nodes: []
    ... ''')
    >>> result["valid"]
    True

    >>> result = validate("invalid: yaml: content")
    >>> result["valid"]
    False
    """
    if lenient:
        return _validate_lenient(yaml_content)
    return _validate_full(yaml_content, base_path=base_path)


def _validate_full(
    yaml_content: str,
    base_path: Path | None = None,
) -> dict[str, Any]:
    """Full validation with YamlPipelineBuilder."""
    try:
        builder = YamlPipelineBuilder(base_path=base_path)
        graph, config = builder.build_from_yaml_string(yaml_content)

        return {
            "valid": True,
            "message": "Pipeline is valid",
            "node_count": len(graph),
            "nodes": [node.name for node in graph.values()],
            "ports": list(config.ports.keys()) if config.ports else [],
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _strip_includes(
    obj: Any,
    warnings: list[str],
    *,
    _counter: list[int] | None = None,
) -> Any:
    """Strip ``{"!include": "..."}`` entries from parsed YAML.

    In node lists, replaces includes with placeholder data_node entries.
    Elsewhere, replaces with None.  Collects warnings for each stripped include.
    """
    if _counter is None:
        _counter = [0]

    if isinstance(obj, dict):
        if "!include" in obj and len(obj) == 1:
            path = obj["!include"]
            warnings.append(f"!include '{path}' skipped during lenient validation")
            return None
        return {k: _strip_includes(v, warnings, _counter=_counter) for k, v in obj.items()}

    if isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, dict) and "!include" in item and len(item) == 1:
                path = item["!include"]
                warnings.append(f"!include '{path}' skipped during lenient validation")
                _counter[0] += 1
                result.append({
                    "kind": "data_node",
                    "metadata": {"name": f"__included_{_counter[0]}__"},
                    "spec": {"data": {}},
                })
            else:
                result.append(_strip_includes(item, warnings, _counter=_counter))
        return result

    return obj


def _validate_lenient(yaml_content: str) -> dict[str, Any]:
    """Structure-only validation using core YamlValidator.

    Delegates to the core validator which handles:
    - YAML syntax validation
    - Manifest structure (kind, metadata, spec)
    - Node structure and dependencies
    - Cycle detection

    ``!include`` directives (dict-key syntax) are stripped and replaced
    with placeholder nodes so that structural validation can proceed
    without requiring access to included files.
    """
    try:
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            return {
                "valid": False,
                "error": "YAML must be a dictionary",
                "error_type": "ParseError",
            }

        # Strip !include directives before validation
        include_warnings: list[str] = []
        parsed = _strip_includes(parsed, include_warnings)

        # Use core validator for structural validation
        validator = YamlValidator()
        report = validator.validate(parsed)

        if not report:
            return {
                "valid": False,
                "error": report.errors[0] if report.errors else "Validation failed",
                "error_type": "ValidationError",
            }

        # Extract node names from spec
        nodes = [
            n.get("metadata", {}).get("name", "unknown")
            for n in parsed.get("spec", {}).get("nodes", [])
        ]

        return {
            "valid": True,
            "message": "Pipeline structure is valid",
            "node_count": len(nodes),
            "nodes": nodes,
            "warnings": include_warnings + report.warnings,
        }

    except yaml.YAMLError as e:
        return {
            "valid": False,
            "error": f"YAML syntax error: {e}",
            "error_type": "YAMLError",
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
