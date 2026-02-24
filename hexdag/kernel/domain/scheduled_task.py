"""Domain model for scheduled pipeline tasks.

Used by :class:`~hexdag.stdlib.lib.scheduler.Scheduler`
to track delayed and recurring pipeline executions.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ScheduleType(StrEnum):
    """Type of schedule."""

    ONCE = "once"
    RECURRING = "recurring"


class TaskStatus(StrEnum):
    """Lifecycle status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ScheduledTask:
    """Record of a scheduled pipeline execution."""

    task_id: str
    pipeline_name: str
    schedule_type: ScheduleType
    initial_input: dict[str, Any] = field(default_factory=dict)
    delay_seconds: float = 0.0
    interval_seconds: float | None = None
    ref_id: str | None = None
    ref_type: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    next_run_at: float | None = None
    last_run_at: float | None = None
    run_count: int = 0
    last_run_id: str | None = None
    error: str | None = None


def scheduled_task_to_storage(task: ScheduledTask) -> dict[str, Any]:
    """Serialise a ScheduledTask to a storage-ready dict."""
    return dataclasses.asdict(task)


def scheduled_task_from_storage(data: dict[str, Any]) -> ScheduledTask:
    """Reconstruct a ScheduledTask from a storage dict."""
    data = dict(data)
    data["schedule_type"] = ScheduleType(data["schedule_type"])
    data["status"] = TaskStatus(data["status"])
    return ScheduledTask(**data)
