"""Execution API for hexdag studio.

Provides test execution of pipelines with per-pipeline environment-based adapters.
Uses the unified hexdag.api layer for feature parity with MCP server.
"""

import json
import sys
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hexdag import api
from hexdag_studio.server.routes.environments import (
    discover_environments_for_pipeline,
    get_environment_ports_with_overrides,
)
from hexdag_studio.server.routes.files import get_workspace_root

router = APIRouter(prefix="/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    """Request to execute a pipeline."""

    content: str
    inputs: dict[str, Any] = {}
    environment: str | None = None  # Environment name (local, dev, prod, etc.)
    pipeline_path: str | None = None  # Path to pipeline for environment discovery
    timeout: float = 30.0


class NodeResult(BaseModel):
    """Result from a single node execution."""

    name: str
    status: str  # completed, failed, skipped
    output: Any | None = None
    error: str | None = None
    duration_ms: float | None = None


class ExecuteResponse(BaseModel):
    """Pipeline execution result."""

    success: bool
    nodes: list[NodeResult]
    final_output: Any | None = None
    error: str | None = None
    duration_ms: float
    environment: str | None = None
    environment_source: str | None = None  # 'inline', 'folder', or 'default'


def _setup_workspace_path() -> None:
    """Add workspace root to sys.path for local module imports."""
    try:
        workspace_root = get_workspace_root()
        # Check if workspace has a parent with tools/adapters (project root pattern)
        # e.g., workspace is "examples/raven/pipelines", project root is "examples/raven"
        project_root = (
            workspace_root.parent if workspace_root.name == "pipelines" else workspace_root
        )
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
    except Exception:
        pass  # Workspace not set, continue without modification


@router.post("", response_model=ExecuteResponse)
async def execute_pipeline(request: ExecuteRequest) -> ExecuteResponse:
    """Execute a pipeline with environment-based adapters.

    Uses the unified hexdag.api.execution module for feature parity with MCP.

    Environment discovery order:
    1. Inline environments in YAML (spec.environments)
    2. environments/ folder relative to pipeline
    3. Default mock environment

    This is for testing/preview purposes. Production execution
    should use the CLI or programmatic API.
    """
    # Add workspace root to sys.path so local modules (like tools/) can be imported
    _setup_workspace_path()

    # Discover environments for this pipeline
    environments, env_source = discover_environments_for_pipeline(
        request.pipeline_path,
        request.content,
    )

    # Determine which environment to use
    env_name = request.environment
    if not env_name:
        # Default to first discovered environment
        env_name = environments[0].name if environments else "local"

    # Get port configuration for selected environment (including node overrides)
    global_ports_config, node_overrides_config = get_environment_ports_with_overrides(
        env_name,
        request.pipeline_path,
        request.content,
    )

    # Create global ports based on environment configuration using unified API
    ports = api.execution.create_ports_from_config(global_ports_config)

    # Create per-node port instances from node overrides
    node_ports: dict[str, dict[str, Any]] = {}
    for node_name, node_port_configs in node_overrides_config.items():
        node_ports[node_name] = api.execution.create_ports_from_config(node_port_configs)

    # Execute using unified API with per-node port support
    result = await api.execution.execute(
        yaml_content=request.content,
        inputs=request.inputs,
        ports=ports,
        node_ports=node_ports if node_ports else None,
        timeout=request.timeout,
    )

    # Convert API result to response format
    node_results: list[NodeResult] = [
        NodeResult(
            name=node_data.get("name", "unknown"),
            status=node_data.get("status", "unknown"),
            output=node_data.get("output"),
            error=node_data.get("error"),
            duration_ms=node_data.get("duration_ms"),
        )
        for node_data in result.get("nodes", [])
    ]

    return ExecuteResponse(
        success=result.get("success", False),
        nodes=node_results,
        final_output=result.get("final_output"),
        error=result.get("error"),
        duration_ms=result.get("duration_ms", 0.0),
        environment=env_name,
        environment_source=env_source,
    )


@router.post("/dry-run")
async def dry_run(request: ExecuteRequest) -> dict[str, Any]:
    """Analyze pipeline without executing.

    Uses the unified hexdag.api.execution module for feature parity with MCP.

    Returns execution plan, dependency order, and estimated complexity.
    """
    # Add workspace root to sys.path for local module imports
    _setup_workspace_path()

    # Use unified API for dry run
    return api.execution.dry_run(request.content, request.inputs)


@router.post("/stream")
async def execute_pipeline_stream(request: ExecuteRequest) -> StreamingResponse:
    """Execute a pipeline with real-time streaming of node status.

    Uses the hexdag observer system to emit events as nodes actually execute,
    providing true real-time updates for the UI.

    Returns Server-Sent Events (SSE) with node execution progress.
    Event types:
    - init: Initial setup info (environment)
    - plan: Execution plan with waves and nodes
    - wave_start: A wave of nodes is starting
    - node_start: A node is starting execution
    - node_complete: A node completed successfully
    - node_failed: A node failed
    - complete: Pipeline execution completed
    - error: An error occurred
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for pipeline execution using real observer-based streaming."""
        _setup_workspace_path()

        try:
            # Discover environments
            environments, env_source = discover_environments_for_pipeline(
                request.pipeline_path,
                request.content,
            )

            env_name = request.environment
            if not env_name:
                env_name = environments[0].name if environments else "local"

            # Get port configuration
            global_ports_config, node_overrides_config = get_environment_ports_with_overrides(
                env_name,
                request.pipeline_path,
                request.content,
            )

            # Send initial event
            yield _sse_event(
                "init",
                {"environment": env_name, "environment_source": env_source},
            )

            # Create ports
            ports = api.execution.create_ports_from_config(global_ports_config)
            node_ports: dict[str, dict[str, Any]] = {}
            for node_name, node_port_configs in node_overrides_config.items():
                node_ports[node_name] = api.execution.create_ports_from_config(node_port_configs)

            # Use the streaming execution API - events come as nodes actually execute
            async for event in api.execution.execute_streaming(
                yaml_content=request.content,
                inputs=request.inputs,
                ports=ports,
                node_ports=node_ports if node_ports else None,
                timeout=request.timeout,
            ):
                event_type = event["event"]
                event_data = event["data"]

                # Add environment info to complete event
                if event_type == "complete":
                    event_data["environment"] = env_name
                    event_data["environment_source"] = env_source

                yield _sse_event(event_type, event_data)

        except Exception as e:
            yield _sse_event(
                "error",
                {"error": str(e), "duration_ms": 0},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format data as a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
