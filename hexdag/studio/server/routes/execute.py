"""Execution API for hexdag studio.

Provides test execution of pipelines with mock adapters.
"""

import asyncio
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    """Request to execute a pipeline."""

    content: str
    inputs: dict[str, Any] = {}
    use_mocks: bool = True
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


@router.post("", response_model=ExecuteResponse)
async def execute_pipeline(request: ExecuteRequest) -> ExecuteResponse:
    """Execute a pipeline with optional mock adapters.

    This is for testing/preview purposes. Production execution
    should use the CLI or programmatic API.
    """
    import time

    start = time.perf_counter()
    node_results: list[NodeResult] = []

    try:
        from hexdag.core.orchestration.orchestrator import Orchestrator
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        # Build pipeline
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(request.content)

        # Create orchestrator with mocks if requested
        if request.use_mocks:
            # Use mock adapters for safe testing
            from hexdag.builtin.adapters.memory import InMemoryMemory
            from hexdag.builtin.adapters.mock import MockLLM

            orchestrator = Orchestrator(
                ports={
                    "llm": MockLLM(),
                    "memory": InMemoryMemory(),
                }
            )
        else:
            orchestrator = Orchestrator()

        # Track node results via events
        node_timings: dict[str, float] = {}

        def on_node_started(event: Any) -> None:
            node_timings[event.node_id] = time.perf_counter()

        def on_node_completed(event: Any) -> None:
            start_time = node_timings.get(event.node_id, start)
            duration = (time.perf_counter() - start_time) * 1000
            node_results.append(
                NodeResult(
                    name=event.node_id,
                    status="completed",
                    output=event.output if hasattr(event, "output") else None,
                    duration_ms=duration,
                )
            )

        def on_node_failed(event: Any) -> None:
            start_time = node_timings.get(event.node_id, start)
            duration = (time.perf_counter() - start_time) * 1000
            node_results.append(
                NodeResult(
                    name=event.node_id,
                    status="failed",
                    error=str(event.error) if hasattr(event, "error") else "Unknown error",
                    duration_ms=duration,
                )
            )

        # Subscribe to events
        orchestrator.on("node_started", on_node_started)
        orchestrator.on("node_completed", on_node_completed)
        orchestrator.on("node_failed", on_node_failed)

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                orchestrator.run(graph, request.inputs),
                timeout=request.timeout,
            )
            success = True
            final_output = result
            error = None
        except TimeoutError:
            success = False
            final_output = None
            error = f"Execution timed out after {request.timeout}s"

    except Exception as e:
        success = False
        final_output = None
        error = str(e)

    duration = (time.perf_counter() - start) * 1000

    return ExecuteResponse(
        success=success,
        nodes=node_results,
        final_output=final_output,
        error=error,
        duration_ms=duration,
    )


@router.post("/dry-run")
async def dry_run(request: ExecuteRequest) -> dict[str, Any]:
    """Analyze pipeline without executing.

    Returns execution plan, dependency order, and estimated complexity.
    """
    try:
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(request.content)

        # Get execution order
        execution_order = list(graph.topological_sort())

        # Analyze dependencies
        dependency_map = {}
        for node_id in execution_order:
            node = graph.get_node(node_id)
            dependency_map[node_id] = {
                "dependencies": list(node.dependencies) if node else [],
                "kind": node.node_type if node and hasattr(node, "node_type") else "unknown",
            }

        return {
            "valid": True,
            "execution_order": execution_order,
            "node_count": len(execution_order),
            "dependency_map": dependency_map,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }
