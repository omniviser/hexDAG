"""Domain model for pipeline run tracking.

Used by :class:`~hexdag.stdlib.lib.process_registry.ProcessRegistry`
to track the state and metadata of pipeline executions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    """Lifecycle status of a pipeline run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class PipelineRun:
    """Immutable-ish record of a single pipeline execution."""

    run_id: str
    pipeline_name: str
    status: RunStatus = RunStatus.CREATED
    parent_run_id: str | None = None
    ref_id: str | None = None
    ref_type: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    node_results: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
