"""Process management API — kernel primitives for multi-pipeline coordination.

Provides 9 unified functions that both the MCP server and hexdag-studio
REST API consume. Each function delegates to the appropriate lib instance.

MCP server usage::

    from hexdag.api import processes

    @mcp.tool()
    async def list_pipeline_runs(ref_id: str | None = None, status: str | None = None):
        return await processes.list_pipeline_runs(registry, ref_id=ref_id, status=status)

Studio REST API::

    @router.get("/processes/runs")
    async def get_runs(ref_id: str | None = None, status: str | None = None):
        return await processes.list_pipeline_runs(registry, ref_id=ref_id, status=status)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel import PipelineSpawner
    from hexdag.stdlib.lib.entity_state import EntityState
    from hexdag.stdlib.lib.process_registry import ProcessRegistry


# ---------------------------------------------------------------------------
# ProcessRegistry tools
# ---------------------------------------------------------------------------


async def list_pipeline_runs(
    registry: ProcessRegistry,
    *,
    ref_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List pipeline runs, optionally filtered by ref_id or status.

    Args
    ----
        registry: ProcessRegistry instance.
        ref_id: Filter by business reference ID.
        status: Filter by run status.
        limit: Maximum results (default 50).

    Returns
    -------
        List of run detail dicts.
    """
    if ref_id is not None:
        return await registry.alist_by_ref(ref_id)
    return await registry.alist(status=status, limit=limit)


async def get_pipeline_run(
    registry: ProcessRegistry,
    run_id: str,
) -> dict[str, Any] | None:
    """Get a single pipeline run by ID.

    Args
    ----
        registry: ProcessRegistry instance.
        run_id: The run ID to look up.

    Returns
    -------
        Run detail dict or None.
    """
    return await registry.aget(run_id)


# ---------------------------------------------------------------------------
# PipelineSpawner tools
# ---------------------------------------------------------------------------


async def spawn_pipeline(
    spawner: PipelineSpawner,
    pipeline_name: str,
    initial_input: dict[str, Any] | None = None,
    *,
    ref_id: str | None = None,
    ref_type: str | None = None,
    wait: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Spawn a new pipeline run.

    Args
    ----
        spawner: PipelineSpawner instance.
        pipeline_name: Name of the pipeline to execute.
        initial_input: Input data for the pipeline.
        ref_id: Business reference ID.
        ref_type: Business reference type.
        wait: If True, block until pipeline completes.
        timeout: Max wait time in seconds.

    Returns
    -------
        Dict with run_id and status.
    """
    run_id = await spawner.aspawn(
        pipeline_name,
        initial_input or {},
        ref_id=ref_id,
        ref_type=ref_type,
        wait=wait,
        timeout=timeout,
    )
    return {"run_id": run_id, "pipeline_name": pipeline_name, "status": "spawned"}


# ---------------------------------------------------------------------------
# EntityState tools
# ---------------------------------------------------------------------------


async def get_entity_state(
    entity_state: EntityState,
    entity_type: str,
    entity_id: str,
) -> dict[str, Any] | None:
    """Get the current state of an entity.

    Args
    ----
        entity_state: EntityState instance.
        entity_type: The entity type (e.g. "order").
        entity_id: The entity ID.

    Returns
    -------
        Dict with entity_type, entity_id, and state, or None.
    """
    return await entity_state.aget_state(entity_type, entity_id)


async def transition_entity(
    entity_state: EntityState,
    entity_type: str,
    entity_id: str,
    to_state: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Transition an entity to a new state.

    Args
    ----
        entity_state: EntityState instance.
        entity_type: The entity type.
        entity_id: The entity ID.
        to_state: Target state.
        reason: Optional reason for the transition.

    Returns
    -------
        Dict with from_state, to_state, and entity details.
    """
    return await entity_state.atransition(entity_type, entity_id, to_state, reason=reason)


async def get_entity_history(
    entity_state: EntityState,
    entity_type: str,
    entity_id: str,
) -> list[dict[str, Any]]:
    """Get the full state transition history of an entity.

    Args
    ----
        entity_state: EntityState instance.
        entity_type: The entity type.
        entity_id: The entity ID.

    Returns
    -------
        List of transition records, oldest first.
    """
    return await entity_state.aget_history(entity_type, entity_id)
