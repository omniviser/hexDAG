"""Validation API for hexdag studio.

Provides real-time validation of YAML pipeline configurations.
Uses the unified hexdag.api layer for feature parity with MCP server.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from hexdag import api

router = APIRouter(prefix="/validate", tags=["validate"])


class ValidationRequest(BaseModel):
    """Request to validate YAML content."""

    content: str
    filename: str | None = None
    lenient: bool = False


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
    warnings: list[str] | None = None


@router.post("", response_model=ValidationResponse)
async def validate_yaml(request: ValidationRequest) -> ValidationResponse:
    """Validate YAML pipeline configuration.

    Uses the unified hexdag.api.validation module for feature parity with MCP.

    Checks:
    - YAML syntax
    - hexdag schema compliance
    - Node dependency validity
    - Type compatibility (full mode only)

    Args:
        request: Validation request with YAML content and options.
            Set lenient=True to skip environment variable resolution (useful for CI/CD).
    """
    # Use unified API for validation
    result = api.validation.validate(request.content, lenient=request.lenient)

    errors: list[ValidationError] = []

    if not result.get("valid", False):
        # Convert API error to ValidationError format
        error_msg = result.get("error", "Unknown validation error")
        errors.append(
            ValidationError(
                message=error_msg,
                severity="error",
            )
        )

    # Get warnings from lenient validation
    warnings = result.get("warnings", [])

    return ValidationResponse(
        valid=result.get("valid", False),
        errors=errors,
        node_count=result.get("node_count"),
        nodes=result.get("nodes"),
        warnings=warnings if warnings else None,
    )


@router.post("/lenient", response_model=ValidationResponse)
async def validate_yaml_lenient(request: ValidationRequest) -> ValidationResponse:
    """Validate YAML pipeline structure without requiring environment variables.

    Use this for CI/CD validation where secrets aren't available.

    Validates:
    - YAML syntax
    - Node structure and dependencies
    - Port configuration format
    - Manifest format (apiVersion, kind, metadata, spec)

    Does NOT validate:
    - Environment variable values
    - Adapter instantiation
    - Module path resolution
    """
    # Force lenient mode
    request.lenient = True
    return await validate_yaml(request)
