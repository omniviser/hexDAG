"""Tests for checkpoint resume: concurrent protection (D) + YAML drift (F)."""

from datetime import UTC, datetime

import pytest

from hexdag.kernel.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.kernel.orchestration.models import CheckpointState
from hexdag.stdlib.adapters.memory.in_memory_memory import InMemoryMemory


def _make_checkpoint(
    run_id: str = "run-1",
    status: str = "saved",
    node_results: dict | None = None,
    graph_snapshot: dict | None = None,
) -> CheckpointState:
    return CheckpointState(
        run_id=run_id,
        dag_id="test-pipeline",
        graph_snapshot=graph_snapshot or {},
        initial_input={"x": 1},
        node_results=node_results or {"a": {"value": 1}},
        completed_node_ids=list((node_results or {"a": {"value": 1}}).keys()),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status=status,
    )


class TestConcurrentResumeProtection:
    """Bug D: Two concurrent resume() calls should not both succeed."""

    @pytest.mark.asyncio()
    async def test_load_for_resume_marks_as_resuming(self):
        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        await mgr.save(_make_checkpoint("run-1"))
        checkpoint = await mgr.load_for_resume("run-1")

        assert checkpoint is not None
        assert checkpoint.status == "resuming"

        # Verify it was persisted
        reloaded = await mgr.load("run-1")
        assert reloaded is not None
        assert reloaded.status == "resuming"

    @pytest.mark.asyncio()
    async def test_second_resume_blocked(self):
        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        await mgr.save(_make_checkpoint("run-1"))

        # First resume succeeds
        checkpoint = await mgr.load_for_resume("run-1")
        assert checkpoint is not None

        # Second resume is blocked
        with pytest.raises(ValueError, match="already being resumed"):
            await mgr.load_for_resume("run-1")

    @pytest.mark.asyncio()
    async def test_load_for_resume_not_found(self):
        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        result = await mgr.load_for_resume("nonexistent")
        assert result is None

    @pytest.mark.asyncio()
    async def test_completed_checkpoint_can_be_resumed(self):
        """A completed checkpoint should be resumable (re-run)."""
        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        await mgr.save(_make_checkpoint("run-1", status="completed"))
        checkpoint = await mgr.load_for_resume("run-1")
        assert checkpoint is not None

    @pytest.mark.asyncio()
    async def test_saved_checkpoint_can_be_resumed(self):
        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        await mgr.save(_make_checkpoint("run-1", status="saved"))
        checkpoint = await mgr.load_for_resume("run-1")
        assert checkpoint is not None


class TestCheckpointStatus:
    """CheckpointState.status field behavior."""

    def test_default_status_is_saved(self):
        state = _make_checkpoint()
        assert state.status == "saved"

    def test_status_serialization_roundtrip(self):
        state = _make_checkpoint(status="resuming")
        json_str = state.model_dump_json()
        restored = CheckpointState.model_validate_json(json_str)
        assert restored.status == "resuming"
