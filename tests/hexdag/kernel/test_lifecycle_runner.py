"""Tests for LifecycleRunner (Phase 3)."""

from unittest.mock import AsyncMock, patch

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


# ---------------------------------------------------------------------------
# Multi-entity namespace collision tests
# ---------------------------------------------------------------------------


def _make_multi_entity_system(**overrides) -> SystemConfig:
    """Two entity types with overlapping state names but different processes."""
    base = {
        "metadata": {"name": "multi-entity-test"},
        "processes": [
            {"name": "investigate_ticket", "pipeline": "pipelines/investigate_ticket.yaml"},
            {"name": "cancel_ticket", "pipeline": "pipelines/cancel_ticket.yaml"},
            {"name": "investigate_order", "pipeline": "pipelines/investigate_order.yaml"},
            {"name": "cancel_order", "pipeline": "pipelines/cancel_order.yaml"},
        ],
        "state_machines": {
            "ticket": {
                "initial": "OPEN",
                "transitions": {
                    "OPEN": ["INVESTIGATING", "CANCELLED"],
                    "INVESTIGATING": ["RESOLVED", "CANCELLED"],
                    "RESOLVED": ["CLOSED"],
                },
            },
            "order": {
                "initial": "PENDING",
                "transitions": {
                    "PENDING": ["INVESTIGATING", "CANCELLED"],
                    "INVESTIGATING": ["FULFILLED", "CANCELLED"],
                    "FULFILLED": ["CLOSED"],
                },
            },
        },
        "states": {
            "INVESTIGATING": {"on_enter": "investigate_ticket"},
            "CANCELLED": {"on_enter": "cancel_ticket", "terminal": True},
            "CLOSED": {"terminal": True},
        },
    }
    base.update(overrides)
    return SystemConfig.model_validate(base)


class TestLifecycleRunnerMultiEntity:
    """Tests for entity-type-qualified state lookups (namespace collision fix)."""

    @pytest.mark.asyncio()
    async def test_same_state_different_entity_types(self):
        """Both ticket and order can enter INVESTIGATING without collision."""
        runner = LifecycleRunner()
        await runner.start(_make_multi_entity_system())
        runner._spawn_process = AsyncMock()  # type: ignore[method-assign]

        await runner.transition("ticket", "T-1", "INVESTIGATING")
        await runner.transition("order", "O-1", "INVESTIGATING")

        assert runner._spawn_process.call_count == 2
        # Both should spawn "investigate_ticket" (the process mapped to INVESTIGATING)
        for call in runner._spawn_process.call_args_list:
            assert call[0][0] == "investigate_ticket"
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_per_entity_type_state_processes(self):
        """Different entity types with per-type state process overrides."""
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [
                {"name": "handle_ticket", "pipeline": "p/a.yaml"},
                {"name": "handle_order", "pipeline": "p/b.yaml"},
            ],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {"OPEN": ["PROCESSING"]},
                },
                "order": {
                    "initial": "NEW",
                    "transitions": {"NEW": ["PROCESSING"]},
                },
            },
            "states": {
                # Both entity types share PROCESSING state but map to same process.
                # The key point: no collision between ticket.PROCESSING and order.PROCESSING
                "PROCESSING": {"on_enter": "handle_ticket"},
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)
        runner._spawn_process = AsyncMock()  # type: ignore[method-assign]

        await runner.transition("ticket", "T-1", "PROCESSING")
        await runner.transition("order", "O-1", "PROCESSING")

        # Both should trigger since PROCESSING is in both state machines
        assert runner._spawn_process.call_count == 2
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_per_entity_terminal_states(self):
        """Terminal state for one entity type doesn't affect another."""
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {
                        "OPEN": ["DONE"],
                        "DONE": ["REOPENED"],
                    },
                },
                "order": {
                    "initial": "NEW",
                    "transitions": {
                        "NEW": ["DONE"],
                    },
                },
            },
            "states": {
                "DONE": {"terminal": True},
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)

        # Both entity types reach DONE — both should be GC'd since
        # DONE is terminal for both (it's in both state machines).
        await runner.transition("ticket", "T-1", "DONE")
        await runner.transition("order", "O-1", "DONE")

        assert runner.active_entities == 0
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_per_entity_requires(self):
        """Required fields can be different per entity type sharing a state name."""
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {"OPEN": ["ACCEPTED"]},
                },
                "order": {
                    "initial": "NEW",
                    "transitions": {"NEW": ["ACCEPTED"]},
                },
            },
            "states": {
                "ACCEPTED": {"requires": ["assignee"]},
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)

        # Both entity types should enforce the requires
        with pytest.raises(LifecycleError, match="requires fields"):
            await runner.transition("ticket", "T-1", "ACCEPTED")

        # With payload it passes
        result = await runner.transition("order", "O-1", "ACCEPTED", payload={"assignee": "user-1"})
        assert result["to_state"] == "ACCEPTED"
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_per_entity_guards(self):
        """Guards are entity-type-qualified — same transition name, different guards."""
        config = SystemConfig.model_validate({
            "metadata": {"name": "test"},
            "processes": [],
            "state_machines": {
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {
                        "OPEN": [
                            {"to": "ESCALATED", "guard": "priority == 'critical'"},
                        ],
                    },
                },
                "order": {
                    "initial": "NEW",
                    "transitions": {
                        "NEW": [
                            {"to": "ESCALATED", "guard": "amount > 1000"},
                        ],
                    },
                },
            },
        })
        runner = LifecycleRunner()
        await runner.start(config)

        # Ticket guard checks priority
        with pytest.raises(InvalidTransitionError, match="Guard blocked"):
            await runner.transition("ticket", "T-1", "ESCALATED", payload={"priority": "low"})

        # Order guard checks amount — ticket guard should NOT apply
        result = await runner.transition("order", "O-1", "ESCALATED", payload={"amount": 5000})
        assert result["to_state"] == "ESCALATED"
        await runner.stop()

    @pytest.mark.asyncio()
    async def test_payload_does_not_shadow_framework_fields(self):
        """Payload key named 'entity_type' should not override framework field."""
        runner = LifecycleRunner()
        await runner.start(_make_multi_entity_system())
        runner._spawn_process = AsyncMock()  # type: ignore[method-assign]

        await runner.transition(
            "ticket",
            "T-1",
            "INVESTIGATING",
            payload={"entity_type": "SHOULD_NOT_OVERRIDE"},
        )

        # Verify spawn was called and input_data has correct entity_type
        call_args = runner._spawn_process.call_args
        request = call_args[0][1]  # TransitionRequest
        assert request.entity_type == "ticket"
        await runner.stop()


class TestLifecycleRunnerEntityStateSharing:
    """Tests that spawned pipelines receive the System's shared EntityState."""

    @pytest.mark.asyncio()
    async def test_spawn_process_passes_shared_entity_state(self):
        """PipelineRunner receives the LifecycleRunner's EntityState via service_overrides."""
        runner = LifecycleRunner()
        config = _make_ticket_system()
        await runner.start(config)

        captured_kwargs: dict = {}

        original_init = __import__(
            "hexdag.kernel.pipeline_runner", fromlist=["PipelineRunner"]
        ).PipelineRunner.__init__

        def capture_init(self_inner, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self_inner, **kwargs)

        with (
            patch(
                "hexdag.kernel.lifecycle_runner.PipelineRunner.__init__",
                capture_init,
            ),
            patch(
                "hexdag.kernel.lifecycle_runner.PipelineRunner.run",
                new_callable=AsyncMock,
            ),
        ):
            await runner.transition("ticket", "T-1", "INVESTIGATING")

        assert "service_overrides" in captured_kwargs
        assert captured_kwargs["service_overrides"]["entity_state"] is runner.entity_state
        await runner.stop()
