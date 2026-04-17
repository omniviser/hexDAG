"""Tests for LifecycleRunner (Phase 3)."""

from unittest.mock import AsyncMock

import pytest

from hexdag.kernel.domain.system_config import SystemConfig
from hexdag.kernel.exceptions import InvalidTransitionError
from hexdag.kernel.lifecycle_runner import (
    CascadeDepthExceeded,
    LifecycleError,
    LifecycleRunner,
)


def _make_ticket_system(**overrides) -> SystemConfig:
    base = {
        "metadata": {"name": "test-system"},
        "processes": [
            {"name": "investigate", "pipeline": "pipelines/investigate.yaml"},
            {"name": "escalate", "pipeline": "pipelines/escalate.yaml"},
        ],
        "state_machines": {
            "ticket": {
                "initial": "OPEN",
                "transitions": {
                    "OPEN": ["INVESTIGATING", "ESCALATED", "CLOSED"],
                    "INVESTIGATING": ["RESOLVED", "ESCALATED"],
                    "ESCALATED": ["INVESTIGATING", "RESOLVED"],
                    "RESOLVED": ["CLOSED"],
                },
            },
        },
        "states": {
            "INVESTIGATING": {"on_enter": "investigate"},
            "ESCALATED": {"on_enter": "escalate"},
            "CLOSED": {"terminal": True},
        },
    }
    base.update(overrides)
    return SystemConfig.model_validate(base)


class TestLifecycleRunnerStartStop:
    @pytest.mark.asyncio()
    async def test_start(self):
        runner = LifecycleRunner()
        config = _make_ticket_system()
        await runner.start(config)
        assert runner.is_running
        assert runner.active_entities == 0
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_double_start_raises(self):
        runner = LifecycleRunner()
        config = _make_ticket_system()
        await runner.start(config)
        with pytest.raises(LifecycleError, match="already started"):
            await runner.start(config)
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_stop_when_not_running(self):
        runner = LifecycleRunner()
        await runner.stop()  # Should not raise

    @pytest.mark.asyncio()
    async def test_transition_before_start_raises(self):
        runner = LifecycleRunner()
        with pytest.raises(LifecycleError, match="not started"):
            await runner.transition("ticket", "T-1", "INVESTIGATING")


class TestLifecycleRunnerTransitions:
    @pytest.fixture()
    async def runner(self):
        r = LifecycleRunner()
        await r.start(_make_ticket_system())
        # Mock pipeline spawning — no real YAML files needed
        r._spawn_process = AsyncMock()  # type: ignore[method-assign]
        yield r
        await r.stop()

    @pytest.mark.asyncio()
    async def test_basic_transition(self, runner):
        result = await runner.transition("ticket", "T-1", "INVESTIGATING")
        assert result["from_state"] == "OPEN"
        assert result["to_state"] == "INVESTIGATING"

    @pytest.mark.asyncio()
    async def test_auto_registers_entity(self, runner):
        assert runner.active_entities == 0
        await runner.transition("ticket", "T-1", "INVESTIGATING")
        assert runner.active_entities == 1

    @pytest.mark.asyncio()
    async def test_spawns_on_enter_process(self, runner):
        await runner.transition("ticket", "T-1", "INVESTIGATING")
        runner._spawn_process.assert_called_once()
        call_args = runner._spawn_process.call_args
        assert call_args[0][0] == "investigate"  # process_name

    @pytest.mark.asyncio()
    async def test_invalid_transition_raises(self, runner):
        with pytest.raises(InvalidTransitionError):
            await runner.transition("ticket", "T-1", "RESOLVED")  # OPEN -> RESOLVED not valid

    @pytest.mark.asyncio()
    async def test_unknown_entity_type_raises(self, runner):
        with pytest.raises(LifecycleError, match="Unknown entity type"):
            await runner.transition("nonexistent", "X-1", "SOMETHING")

    @pytest.mark.asyncio()
    async def test_multi_step_transition(self, runner):
        await runner.transition("ticket", "T-1", "INVESTIGATING")
        result = await runner.transition("ticket", "T-1", "RESOLVED")
        assert result["from_state"] == "INVESTIGATING"
        assert result["to_state"] == "RESOLVED"

    @pytest.mark.asyncio()
    async def test_multiple_entities(self, runner):
        await runner.transition("ticket", "T-1", "INVESTIGATING")
        await runner.transition("ticket", "T-2", "ESCALATED")
        assert runner.active_entities == 2

    @pytest.mark.asyncio()
    async def test_draining_rejects_new_transitions(self, runner):
        runner._draining = True
        with pytest.raises(LifecycleError, match="shutting down"):
            await runner.transition("ticket", "T-1", "INVESTIGATING")


class TestLifecycleRunnerGuards:
    @staticmethod
    def _guarded_config():
        return SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {
                        "OPEN": [
                            {"to": "ESCALATED", "guard": "priority == 'critical'"},
                            "INVESTIGATING",
                        ],
                        "INVESTIGATING": ["RESOLVED"],
                        "ESCALATED": ["RESOLVED"],
                    },
                },
            },
        })

    @pytest.mark.asyncio()
    async def test_guard_allows(self):
        runner = LifecycleRunner()
        await runner.start(self._guarded_config())

        result = await runner.transition(
            "ticket",
            "T-1",
            "ESCALATED",
            payload={"priority": "critical"},
        )
        assert result["to_state"] == "ESCALATED"
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_guard_blocks(self):
        runner = LifecycleRunner()
        await runner.start(self._guarded_config())

        with pytest.raises(InvalidTransitionError, match="Guard blocked"):
            await runner.transition(
                "ticket",
                "T-1",
                "ESCALATED",
                payload={"priority": "low"},
            )
        await runner.stop()


class TestLifecycleRunnerCascadeDepth:
    @pytest.mark.asyncio()
    async def test_cascade_depth_exceeded(self):
        runner = LifecycleRunner(max_cascade_depth=2)
        config = _make_ticket_system()
        await runner.start(config)

        # Simulate deep cascade by manually setting cascade depth
        from hexdag.kernel.lifecycle_runner import TransitionRequest

        request = TransitionRequest(
            entity_type="ticket",
            entity_id="T-1",
            to_state="INVESTIGATING",
            cascade_depth=3,
        )

        # Auto-register first
        await runner._register_entity("ticket", "T-1")

        with pytest.raises(CascadeDepthExceeded):
            await runner._execute_transition(request)
        await runner.stop()


class TestLifecycleRunnerGC:
    @pytest.mark.asyncio()
    async def test_terminal_state_gc(self):
        runner = LifecycleRunner()
        await runner.start(_make_ticket_system())
        runner._spawn_process = AsyncMock()  # type: ignore[method-assign]

        await runner.transition("ticket", "T-1", "INVESTIGATING")
        assert runner.active_entities == 1

        await runner.transition("ticket", "T-1", "RESOLVED")
        await runner.transition("ticket", "T-1", "CLOSED")

        # CLOSED is terminal — entity should be GC'd
        assert runner.active_entities == 0
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_gc_cleans_entity_state(self):
        runner = LifecycleRunner()
        await runner.start(_make_ticket_system())
        runner._spawn_process = AsyncMock()  # type: ignore[method-assign]

        await runner.transition("ticket", "T-1", "INVESTIGATING")
        await runner.transition("ticket", "T-1", "RESOLVED")
        await runner.transition("ticket", "T-1", "CLOSED")

        # EntityState should also be cleaned
        assert runner.entity_state is not None
        state = await runner.entity_state.aget_state("ticket", "T-1")
        assert state is None  # GC'd from in-memory
        await runner.stop()


class TestLifecycleRunnerStateDataContracts:
    @pytest.mark.asyncio()
    async def test_requires_validates_payload(self):
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {
                        "OPEN": ["ACCEPTED"],
                    },
                },
            },
            "states": {
                "ACCEPTED": {
                    "requires": ["carrier_id", "rate"],
                },
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)

        # Missing required fields
        with pytest.raises(LifecycleError, match="requires fields"):
            await runner.transition("ticket", "T-1", "ACCEPTED")
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_requires_passes_with_payload(self):
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {
                        "OPEN": ["ACCEPTED"],
                    },
                },
            },
            "states": {
                "ACCEPTED": {
                    "requires": ["carrier_id"],
                },
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)

        result = await runner.transition(
            "ticket",
            "T-1",
            "ACCEPTED",
            payload={"carrier_id": "C-1"},
        )
        assert result["to_state"] == "ACCEPTED"
        await runner.stop()
