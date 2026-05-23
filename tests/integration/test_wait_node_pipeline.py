"""Integration tests for WaitNode suspend/resume pipeline lifecycle."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.kernel.orchestration.event_correlation import (
    EventCorrelationRegistry,
    WaitRegistration,
)
from hexdag.kernel.orchestration.suspension import Suspended
from hexdag.stdlib.adapters.memory.in_memory_memory import InMemoryMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _send_email(inputs: dict, **kwargs) -> dict:
    """Simulate sending an email — passes through conversation_id."""
    return {
        "sent": True,
        "to": inputs.get("to", "carrier@example.com"),
        "conversation_id": inputs.get("conversation_id", "test-conv"),
    }


async def _wait_for_reply(inputs: dict, **kwargs) -> Suspended:
    """Simulate a wait node — returns Suspended signal.

    When this node has a single dependency (send_email), inputs IS
    the send_email output dict.
    """
    conv_id = inputs.get("conversation_id", "test-conv")
    return Suspended(
        event_key=f"email_reply:{conv_id}",
        timeout_seconds=86400,
        setup_result={"waiting_for": "carrier reply"},
    )


async def _analyze_reply(inputs: dict, **kwargs) -> dict:
    """Analyze the carrier's reply (runs after resume).

    When this node has a single dependency (await_reply), inputs IS
    the await_reply output — which on resume is the event_data dict.
    """
    return {
        "action": "accept" if inputs.get("accepted") else "counter",
        "reply_body": inputs.get("body", ""),
    }


def _build_test_graph() -> DirectedGraph:
    """Build a simple 3-node pipeline: send → wait → analyze."""
    graph = DirectedGraph()
    graph.add(
        NodeSpec(
            name="send_email",
            fn=_send_email,
            deps=frozenset(),
            params={},
            literals={},
        )
    )
    graph.add(
        NodeSpec(
            name="await_reply",
            fn=_wait_for_reply,
            deps=frozenset(["send_email"]),
            params={},
            literals={},
        )
    )
    graph.add(
        NodeSpec(
            name="analyze_reply",
            fn=_analyze_reply,
            deps=frozenset(["await_reply"]),
            params={},
            literals={},
        )
    )
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWaitNodeSuspension:
    """Test that a pipeline with a Suspended return value suspends correctly."""

    @pytest.mark.asyncio()
    async def test_orchestrator_suspends_on_suspended_return(self) -> None:
        """Pipeline stops when a node returns Suspended."""
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        graph = _build_test_graph()

        result = await orchestrator.run(
            graph,
            {"conversation_id": "conv-123", "to": "carrier@example.com"},
        )

        # send_email should have completed
        assert "send_email" in result
        assert result["send_email"]["sent"] is True

        # await_reply's setup_result should be stored
        assert "await_reply" in result
        assert result["await_reply"] == {"waiting_for": "carrier reply"}

        # analyze_reply should NOT have run (it depends on await_reply which suspended)
        assert "analyze_reply" not in result

        # Suspension metadata should be present
        assert "_hexdag_suspended" in result
        suspend_meta = result["_hexdag_suspended"]
        assert suspend_meta["event_key"] == "email_reply:conv-123"
        assert suspend_meta["wait_node_name"] == "await_reply"

    @pytest.mark.asyncio()
    async def test_resume_with_event_data(self) -> None:
        """Pipeline resumes correctly with event data injected as wait node output."""
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        graph = _build_test_graph()

        # First run — suspends at await_reply
        result = await orchestrator.run(
            graph,
            {"conversation_id": "conv-123"},
        )
        assert "_hexdag_suspended" in result

        # Simulate resume: pre-seed completed nodes + event data
        pre_seeded = {k: v for k, v in result.items() if not k.startswith("_hexdag_")}
        # Inject event data as the wait node's output
        event_data = {
            "body": "We accept at $2500",
            "accepted": True,
            "from": "carrier@example.com",
        }
        pre_seeded["await_reply"] = event_data

        # Resume — analyze_reply runs with the event data
        resumed_result = await orchestrator.run(
            graph,
            {"conversation_id": "conv-123"},
            pre_seeded_results=pre_seeded,
        )

        # analyze_reply should have executed (inputs = event_data since single dep)
        assert "analyze_reply" in resumed_result
        assert resumed_result["analyze_reply"]["action"] == "accept"
        assert resumed_result["analyze_reply"]["reply_body"] == "We accept at $2500"

        # No suspension this time
        assert "_hexdag_suspended" not in resumed_result


class TestEventCorrelationRegistry:
    """Test the event correlation registry."""

    @pytest.mark.asyncio()
    async def test_register_and_lookup(self) -> None:
        storage = InMemoryMemory()
        registry = EventCorrelationRegistry(storage=storage)

        reg = WaitRegistration(
            run_id="run-abc",
            pipeline_path="negotiation.yaml",
            wait_node_name="await_reply",
            event_key="email_reply:conv-123",
        )
        await registry.register(reg)

        found = await registry.lookup("email_reply:conv-123")
        assert found is not None
        assert found.run_id == "run-abc"
        assert found.pipeline_path == "negotiation.yaml"
        assert found.wait_node_name == "await_reply"

    @pytest.mark.asyncio()
    async def test_lookup_missing_returns_none(self) -> None:
        storage = InMemoryMemory()
        registry = EventCorrelationRegistry(storage=storage)

        assert await registry.lookup("nonexistent") is None

    @pytest.mark.asyncio()
    async def test_remove(self) -> None:
        storage = InMemoryMemory()
        registry = EventCorrelationRegistry(storage=storage)

        reg = WaitRegistration(
            run_id="run-abc",
            pipeline_path="p.yaml",
            wait_node_name="wait",
            event_key="event:1",
        )
        await registry.register(reg)
        assert await registry.remove("event:1") is True
        assert await registry.lookup("event:1") is None
        assert await registry.remove("event:1") is False

    @pytest.mark.asyncio()
    async def test_list_active(self) -> None:
        storage = InMemoryMemory()
        registry = EventCorrelationRegistry(storage=storage)

        for i in range(3):
            await registry.register(
                WaitRegistration(
                    run_id=f"run-{i}",
                    pipeline_path="p.yaml",
                    wait_node_name="wait",
                    event_key=f"event:{i}",
                )
            )

        active = await registry.list_active()
        assert len(active) == 3

    @pytest.mark.asyncio()
    async def test_get_expired(self) -> None:
        import time

        storage = InMemoryMemory()
        registry = EventCorrelationRegistry(storage=storage)

        # Expired registration
        await registry.register(
            WaitRegistration(
                run_id="run-expired",
                pipeline_path="p.yaml",
                wait_node_name="wait",
                event_key="event:expired",
                timeout_at=time.time() - 100,  # Already expired
            )
        )
        # Active registration (no timeout)
        await registry.register(
            WaitRegistration(
                run_id="run-active",
                pipeline_path="p.yaml",
                wait_node_name="wait",
                event_key="event:active",
            )
        )
        # Future registration
        await registry.register(
            WaitRegistration(
                run_id="run-future",
                pipeline_path="p.yaml",
                wait_node_name="wait",
                event_key="event:future",
                timeout_at=time.time() + 86400,
            )
        )

        expired = await registry.get_expired()
        assert len(expired) == 1
        assert expired[0].run_id == "run-expired"


class TestCheckpointWithSuspension:
    """Test that checkpoint correctly stores suspended state."""

    @pytest.mark.asyncio()
    async def test_checkpoint_saves_wait_metadata(self) -> None:
        """Verify checkpoint includes wait metadata when saved with suspended status."""
        from datetime import UTC, datetime

        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)

        from hexdag.kernel.orchestration.models import CheckpointState

        state = CheckpointState(
            run_id="run-suspended-1",
            dag_id="negotiation",
            graph_snapshot={},
            initial_input={"conversation_id": "conv-123"},
            node_results={"send_email": {"sent": True}},
            completed_node_ids=["send_email"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata={
                "pipeline_status": "suspended",
                "wait": {
                    "wait_node_name": "await_reply",
                    "event_key": "email_reply:conv-123",
                    "timeout_seconds": 86400,
                },
            },
            status="suspended",
        )
        await mgr.save(state)

        loaded = await mgr.load_for_resume("run-suspended-1")
        assert loaded is not None
        assert loaded.status == "resuming"  # load_for_resume marks as resuming
        assert loaded.metadata["wait"]["event_key"] == "email_reply:conv-123"
        assert loaded.metadata["wait"]["wait_node_name"] == "await_reply"
