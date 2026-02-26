"""Observer that bridges pipeline events to ProcessRegistry.

Register this observer with the ObserverManager to automatically
populate the ProcessRegistry as pipelines start, complete, and cancel.

Usage::

    from hexdag.stdlib.lib.process_registry import ProcessRegistry
    from hexdag.stdlib.lib.process_registry_observer import ProcessRegistryObserver

    registry = ProcessRegistry()
    observer = ProcessRegistryObserver(registry)
    observer_manager.register(
        observer,
        event_types=PROCESS_REGISTRY_EVENTS,
    )
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
from hexdag.kernel.orchestration.events.events import (
    Event,
    NodeFailed,
    PipelineCompleted,
    PipelineStarted,
)

if TYPE_CHECKING:
    from hexdag.stdlib.lib.process_registry import ProcessRegistry

# Event types this observer cares about
PROCESS_REGISTRY_EVENTS = (
    PipelineStarted,
    PipelineCompleted,
    NodeFailed,
)


class ProcessRegistryObserver:
    """Observer that populates ProcessRegistry from pipeline events.

    Since the event system doesn't carry ``run_id``, this observer
    generates a unique run_id on ``PipelineStarted`` and correlates
    subsequent events by pipeline ``name``.
    """

    def __init__(self, registry: ProcessRegistry) -> None:
        self._registry = registry
        # Maps pipeline name → current run_id (for correlating events)
        self._active_runs: dict[str, str] = {}

    async def handle(self, event: Event) -> None:
        """Handle a pipeline lifecycle event."""
        if isinstance(event, PipelineStarted):
            await self._on_started(event)
        elif isinstance(event, PipelineCompleted):
            if event.status == "cancelled":
                await self._on_cancelled(event)
            else:
                await self._on_completed(event)
        elif isinstance(event, NodeFailed):
            await self._on_node_failed(event)

    async def _on_started(self, event: PipelineStarted) -> None:
        run_id = str(uuid4())
        self._active_runs[event.name] = run_id
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=event.name,
            status=RunStatus.RUNNING,
            started_at=time.time(),
        )
        await self._registry.register(run)

    async def _on_completed(self, event: PipelineCompleted) -> None:
        run_id = self._active_runs.pop(event.name, None)
        if run_id is None:
            return
        await self._registry.update_status(
            run_id,
            RunStatus.COMPLETED,
            duration_ms=event.duration_ms,
            node_results=event.node_results,
            completed_at=time.time(),
        )

    async def _on_node_failed(self, event: NodeFailed) -> None:
        # NodeFailed has `name` (node name), not pipeline name.
        # We need to search active runs — for now, mark the first
        # active run as failed since events don't carry pipeline context.
        # In multi-pipeline scenarios, this is resolved by using
        # per-pipeline observer instances.
        for _pipeline_name, run_id in list(self._active_runs.items()):
            await self._registry.update_status(
                run_id,
                RunStatus.FAILED,
                error=f"Node '{event.name}' failed: {event.error}",
                completed_at=time.time(),
            )
            break

    async def _on_cancelled(self, event: PipelineCompleted) -> None:
        run_id = self._active_runs.pop(event.name, None)
        if run_id is None:
            return
        await self._registry.update_status(
            run_id,
            RunStatus.CANCELLED,
            completed_at=time.time(),
        )

    def get_active_run_id(self, pipeline_name: str) -> str | None:
        """Get the current run_id for a pipeline (utility for testing)."""
        return self._active_runs.get(pipeline_name)
