"""Pipeline validation API.

A thin JSON adapter over the compiler's single front door,
:func:`hexdag.compiler.staged.compile`. All validation logic lives in the
compiler; this module only shapes results for API/MCP/Studio consumers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


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
        If True, validate structure only without requiring environment
        variables or included files (servers, CI). Default: False
    base_path : Path | None
        Base directory for resolving ``!include`` directives.

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - Whether the pipeline is valid
        - message: str - Success message if valid
        - error: str - First error message if invalid
        - error_type: str - Error classification if invalid
        - node_count: int - Number of nodes (if determinable)
        - nodes: list[str] - Node names (if determinable)
        - ports: list[str] - Port names (full validation only)
        - warnings: list[str] - Validation warnings
        - diagnostics: list[dict] - Structured diagnostics
          (code, severity, message, hint, file, line, column)

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
    from hexdag.compiler.staged import compile as compile_pipeline

    if lenient:
        result = compile_pipeline(yaml_content, lenient=True)
    else:
        # Full validation answers "does it build" — run build mode without
        # raising so failures come back as diagnostics.
        result = compile_pipeline(
            yaml_content, mode="build", base_path=base_path, raise_on_error=False
        )

    payload: dict[str, Any] = {
        "valid": result.ok,
        "node_count": len(result.node_names),
        "nodes": result.node_names,
        "warnings": result.warnings,
        "diagnostics": [d.to_dict() for d in result.diagnostics],
    }

    if result.ok:
        payload["message"] = "Pipeline structure is valid" if lenient else "Pipeline is valid"
        if result.config is not None:
            payload["ports"] = list(result.config.ports.keys()) if result.config.ports else []
    else:
        payload["error"] = result.errors[0] if result.errors else "Validation failed"
        payload["error_type"] = result.error_type or "ValidationError"

    return payload
