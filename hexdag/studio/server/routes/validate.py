"""Validation API for hexdag studio.

Provides real-time validation of YAML pipeline configurations.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/validate", tags=["validate"])


class ValidationRequest(BaseModel):
    """Request to validate YAML content."""

    content: str
    filename: str | None = None


class ValidationError(BaseModel):
    """A single validation error."""

    line: int | None = None
    column: int | None = None
    message: str
    severity: str = "error"  # error, warning, info


class ValidationResponse(BaseModel):
    """Validation result."""

    valid: bool
    errors: list[ValidationError]
    node_count: int | None = None
    nodes: list[str] | None = None


@router.post("", response_model=ValidationResponse)
async def validate_yaml(request: ValidationRequest) -> ValidationResponse:
    """Validate YAML pipeline configuration.

    Checks:
    - YAML syntax
    - hexdag schema compliance
    - Node dependency validity
    - Type compatibility
    """
    import yaml

    errors: list[ValidationError] = []

    # Step 1: Parse YAML syntax
    try:
        parsed = yaml.safe_load(request.content)
    except yaml.YAMLError as e:
        line = None
        column = None
        if hasattr(e, "problem_mark") and e.problem_mark:
            line = e.problem_mark.line + 1
            column = e.problem_mark.column + 1

        return ValidationResponse(
            valid=False,
            errors=[
                ValidationError(
                    line=line,
                    column=column,
                    message=f"YAML syntax error: {e}",
                    severity="error",
                )
            ],
        )

    if parsed is None:
        return ValidationResponse(
            valid=False,
            errors=[ValidationError(message="Empty YAML document", severity="error")],
        )

    if not isinstance(parsed, dict):
        return ValidationResponse(
            valid=False,
            errors=[ValidationError(message="YAML root must be a mapping", severity="error")],
        )

    # Step 2: Check hexdag schema
    api_version = parsed.get("apiVersion")
    if not api_version:
        errors.append(
            ValidationError(
                message="Missing 'apiVersion' field (expected: hexdag/v1)",
                severity="error",
            )
        )
    elif api_version != "hexdag/v1":
        errors.append(
            ValidationError(
                message=f"Unknown apiVersion: {api_version} (expected: hexdag/v1)",
                severity="warning",
            )
        )

    kind = parsed.get("kind")
    if not kind:
        errors.append(ValidationError(message="Missing 'kind' field", severity="error"))
    elif kind != "Pipeline":
        errors.append(
            ValidationError(
                message=f"Unknown kind: {kind} (expected: Pipeline)",
                severity="warning",
            )
        )

    metadata = parsed.get("metadata", {})
    if not metadata.get("name"):
        errors.append(
            ValidationError(
                message="Missing 'metadata.name' field",
                severity="error",
            )
        )

    spec = parsed.get("spec", {})
    nodes = spec.get("nodes", [])

    if not nodes:
        errors.append(
            ValidationError(
                message="No nodes defined in 'spec.nodes'",
                severity="warning",
            )
        )

    # Step 3: Validate nodes
    node_names: set[str] = set()
    node_list: list[str] = []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(
                ValidationError(
                    message=f"Node {i} must be a mapping",
                    severity="error",
                )
            )
            continue

        # Check node kind
        node_kind = node.get("kind")
        if not node_kind:
            errors.append(
                ValidationError(
                    message=f"Node {i}: missing 'kind' field",
                    severity="error",
                )
            )

        # Check node name
        node_metadata = node.get("metadata", {})
        node_name = node_metadata.get("name")
        if not node_name:
            errors.append(
                ValidationError(
                    message=f"Node {i}: missing 'metadata.name' field",
                    severity="error",
                )
            )
        else:
            if node_name in node_names:
                errors.append(
                    ValidationError(
                        message=f"Duplicate node name: {node_name}",
                        severity="error",
                    )
                )
            node_names.add(node_name)
            node_list.append(node_name)

        # Check dependencies reference valid nodes
        dependencies = node.get("dependencies", [])
        for dep in dependencies:
            if dep not in node_names:
                # Might be forward reference - check at end
                pass

    # Step 4: Validate all dependencies exist
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_name = node.get("metadata", {}).get("name", "unknown")
        dependencies = node.get("dependencies", [])
        errors.extend(
            ValidationError(
                message=f"Node '{node_name}': unknown dependency '{dep}'",
                severity="error",
            )
            for dep in dependencies
            if dep not in node_names
        )

    # Step 5: Try full hexdag validation if no critical errors
    if not any(e.severity == "error" for e in errors):
        try:
            from hexdag.compiler import YamlPipelineBuilder

            builder = YamlPipelineBuilder()
            builder.build_from_yaml_string(request.content)
        except Exception as e:
            errors.append(
                ValidationError(
                    message=f"Pipeline build error: {e}",
                    severity="error",
                )
            )

    return ValidationResponse(
        valid=not any(e.severity == "error" for e in errors),
        errors=errors,
        node_count=len(node_list),
        nodes=node_list if node_list else None,
    )
