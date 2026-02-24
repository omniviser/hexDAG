"""Round-trip serialization tests for domain model storage helpers."""

from __future__ import annotations

from hexdag.kernel.domain.entity_state import (
    StateTransition,
    state_transition_from_storage,
    state_transition_to_storage,
)
from hexdag.kernel.domain.pipeline_run import (
    PipelineRun,
    RunStatus,
    pipeline_run_from_storage,
    pipeline_run_to_storage,
)
from hexdag.kernel.domain.scheduled_task import (
    ScheduledTask,
    ScheduleType,
    TaskStatus,
    scheduled_task_from_storage,
    scheduled_task_to_storage,
)


class TestPipelineRunRoundTrip:
    def test_minimal(self) -> None:
        run = PipelineRun(run_id="r1", pipeline_name="p1")
        data = pipeline_run_to_storage(run)
        restored = pipeline_run_from_storage(data)
        assert restored.run_id == run.run_id
        assert restored.pipeline_name == run.pipeline_name
        assert restored.status == RunStatus.CREATED

    def test_full_fields(self) -> None:
        run = PipelineRun(
            run_id="r2",
            pipeline_name="p2",
            status=RunStatus.COMPLETED,
            parent_run_id="parent-1",
            ref_id="order-99",
            ref_type="order",
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1005.0,
            duration_ms=4000.0,
            node_results={"n1": "ok"},
            error=None,
            metadata={"env": "prod"},
        )
        data = pipeline_run_to_storage(run)
        restored = pipeline_run_from_storage(data)
        assert restored.status == RunStatus.COMPLETED
        assert restored.ref_id == "order-99"
        assert restored.node_results == {"n1": "ok"}
        assert restored.metadata == {"env": "prod"}

    def test_status_is_str_enum(self) -> None:
        run = PipelineRun(run_id="r3", pipeline_name="p3", status=RunStatus.FAILED)
        data = pipeline_run_to_storage(run)
        # In storage, status is a string
        assert data["status"] == "failed"
        restored = pipeline_run_from_storage(data)
        assert isinstance(restored.status, RunStatus)
        assert restored.status == RunStatus.FAILED


class TestScheduledTaskRoundTrip:
    def test_once_task(self) -> None:
        task = ScheduledTask(
            task_id="t1",
            pipeline_name="p1",
            schedule_type=ScheduleType.ONCE,
            delay_seconds=30.0,
        )
        data = scheduled_task_to_storage(task)
        restored = scheduled_task_from_storage(data)
        assert restored.task_id == "t1"
        assert restored.schedule_type == ScheduleType.ONCE
        assert restored.status == TaskStatus.PENDING
        assert restored.delay_seconds == 30.0

    def test_recurring_task(self) -> None:
        task = ScheduledTask(
            task_id="t2",
            pipeline_name="p2",
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=300.0,
            status=TaskStatus.RUNNING,
            run_count=5,
        )
        data = scheduled_task_to_storage(task)
        restored = scheduled_task_from_storage(data)
        assert restored.schedule_type == ScheduleType.RECURRING
        assert restored.status == TaskStatus.RUNNING
        assert restored.run_count == 5

    def test_enum_coercion(self) -> None:
        task = ScheduledTask(
            task_id="t3",
            pipeline_name="p3",
            schedule_type=ScheduleType.ONCE,
            status=TaskStatus.CANCELLED,
        )
        data = scheduled_task_to_storage(task)
        assert data["schedule_type"] == "once"
        assert data["status"] == "cancelled"
        restored = scheduled_task_from_storage(data)
        assert isinstance(restored.schedule_type, ScheduleType)
        assert isinstance(restored.status, TaskStatus)


class TestStateTransitionRoundTrip:
    def test_initial_transition(self) -> None:
        t = StateTransition(
            entity_type="order",
            entity_id="o-1",
            from_state=None,
            to_state="new",
        )
        data = state_transition_to_storage(t)
        restored = state_transition_from_storage(data)
        assert restored.entity_type == "order"
        assert restored.from_state is None
        assert restored.to_state == "new"

    def test_with_metadata(self) -> None:
        t = StateTransition(
            entity_type="order",
            entity_id="o-1",
            from_state="new",
            to_state="processing",
            metadata={"reason": "payment received"},
        )
        data = state_transition_to_storage(t)
        restored = state_transition_from_storage(data)
        assert restored.metadata == {"reason": "payment received"}

    def test_timestamp_preserved(self) -> None:
        t = StateTransition(
            entity_type="order",
            entity_id="o-1",
            from_state="a",
            to_state="b",
            timestamp=1234567890.123,
        )
        data = state_transition_to_storage(t)
        restored = state_transition_from_storage(data)
        assert restored.timestamp == 1234567890.123
