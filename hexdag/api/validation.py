"""Pipeline validation API.

Provides unified functions for validating hexDAG YAML pipelines.
Delegates to core YamlValidator for structural validation.
"""

from __future__ import annotations

from typing import Any

import yaml

from hexdag.kernel.pipeline_builder import YamlPipelineBuilder
from hexdag.kernel.pipeline_builder.yaml_validator import YamlValidator


def validate(yaml_content: str, lenient: bool = False) -> dict[str, Any]:
    """Validate a YAML pipeline configuration.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    lenient : bool
        If True, validate structure only without requiring environment variables.
        Useful for CI/CD validation where secrets aren't available.
        Default: False

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
    return _validate_full(yaml_content)


def _validate_full(yaml_content: str) -> dict[str, Any]:
    """Full validation with YamlPipelineBuilder."""
    try:
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        return {
            "valid": True,
            "message": "Pipeline is valid",
            "node_count": len(graph.nodes),
            "nodes": [node.name for node in graph.nodes.values()],
            "ports": list(config.ports.keys()) if config.ports else [],
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _validate_lenient(yaml_content: str) -> dict[str, Any]:
    """Structure-only validation using core YamlValidator.

    Delegates to the core validator which handles:
    - YAML syntax validation
    - Manifest structure (kind, metadata, spec)
    - Node structure and dependencies
    - Cycle detection
    """
    try:
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            return {
                "valid": False,
                "error": "YAML must be a dictionary",
                "error_type": "ParseError",
            }

        # Use core validator for structural validation
        validator = YamlValidator()
        report = validator.validate(parsed)

        if not report.is_valid:
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
            "warnings": report.warnings,
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
