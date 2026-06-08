"""System management API — unified functions for ``kind: System`` operations.

Provides functions that both the MCP server and hexdag-studio REST API
consume.  Each function takes a :class:`System` instance as its first
argument and delegates to the appropriate method.

MCP server usage::

    from hexdag.api import systems

    @mcp.tool()
    async def run_process(process_name: str, input_data: dict | None = None):
        return await systems.run_process(system, process_name, input_data)

Studio REST API::

    @router.post("/system/processes/{name}/run")
    async def run_process(name: str, body: RunInput):
        return await systems.run_process(system, name, body.input_data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel.domain.pipeline_result import PipelineResult
    from hexdag.kernel.system import System


# ---------------------------------------------------------------------------
# System execution
# ---------------------------------------------------------------------------


async def run_system(
    system: System,
    input_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute all processes in topological order (DAG mode).

    Args:
        system: System instance.
        input_data: Initial input data passed to root processes.

    Returns:
        Results keyed by process name.
    """
    return await system.run(input_data)


async def run_process(
    system: System,
    process_name: str,
    input_data: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run a single named process (works in both DAG and lifecycle mode).

    Args:
        system: System instance.
        process_name: Name of the process to run.
        input_data: Input data for the pipeline.

    Returns:
        PipelineResult from the pipeline execution.
    """
    return await system.run_process(process_name, input_data)


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------


async def transition_entity(
    system: System,
    entity_type: str,
    entity_id: str,
    to_state: str,
    *,
    reason: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request an entity state transition (lifecycle mode).

    Args:
        system: System instance.
        entity_type: Entity type (must match a registered state machine).
        entity_id: Unique entity identifier.
        to_state: Target state.
        reason: Optional reason for the transition.
        payload: Additional data to pass to the triggered pipeline.

    Returns:
        Transition result dict.
    """
    return await system.transition(
        entity_type,
        entity_id,
        to_state,
        reason=reason,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def list_processes(system: System) -> list[dict[str, Any]]:
    """List all processes declared in the system.

    Args:
        system: System instance.

    Returns:
        List of process info dicts.
    """
    return [
        {
            "name": p.name,
            "pipeline": p.pipeline,
            "input_schema": p.input_schema,
            "output_schema": p.output_schema,
        }
        for p in system.config.processes
    ]


async def get_entity_state(
    system: System,
    entity_type: str,
    entity_id: str,
) -> dict[str, Any] | None:
    """Get the current state of an entity (lifecycle mode).

    Args:
        system: System instance.
        entity_type: The entity type.
        entity_id: The entity ID.

    Returns:
        Dict with entity state info, or None if not tracked.
    """
    es = system.entity_state
    if es is None:
        return None
    return await es.aget_state(entity_type, entity_id)
