"""Tests for entity lifecycle events (Phase 1)."""

from hexdag.kernel.orchestration.events.events import (
    EntityCompensationEvent,
    EntityGarbageCollected,
    EntityObligationFailed,
    StateTransitionEvent,
    TransitionContext,
)


class TestTransitionContext:
    def test_frozen(self):
        ctx = TransitionContext(run_id="r1", pipeline_name="p1", node_name="n1")
        assert ctx.run_id == "r1"
        assert ctx.pipeline_name == "p1"
        assert ctx.node_name == "n1"


class TestStateTransitionEvent:
    def test_basic_fields(self):
        event = StateTransitionEvent(
            entity_type="ticket",
            entity_id="T-1",
            from_state="OPEN",
            to_state="INVESTIGATING",
            reason="Agent started",
            node_name="investigate",
        )
        assert event.entity_type == "ticket"
        assert event.from_state == "OPEN"
        assert event.to_state == "INVESTIGATING"
        assert event.timestamp is not None

    def test_log_message(self):
        event = StateTransitionEvent(
            entity_type="ticket",
            entity_id="T-1",
            from_state="OPEN",
            to_state="INVESTIGATING",
            reason="test",
        )
        msg = event.log_message()
        assert "ticket:T-1" in msg
        assert "OPEN" in msg
        assert "INVESTIGATING" in msg

    def test_optional_fields(self):
        event = StateTransitionEvent(
            entity_type="load",
            entity_id="L-1",
            from_state="PENDING",
            to_state="POSTED",
        )
        assert event.reason is None
        assert event.node_name == ""


class TestEntityGarbageCollected:
    def test_fields(self):
        event = EntityGarbageCollected(
            entity_type="ticket",
            entity_id="T-1",
            final_state="CLOSED",
            lifetime_ms=5000.0,
            transition_count=4,
        )
        assert event.final_state == "CLOSED"
        assert event.transition_count == 4
        assert "CLOSED" in event.log_message()


class TestEntityObligationFailed:
    def test_fields(self):
        event = EntityObligationFailed(
            entity_type="load",
            entity_id="L-1",
            state="ACCEPTED",
            obligation="pipeline",
            pipeline_run_id="run-123",
            error="Pipeline timed out",
        )
        assert event.obligation == "pipeline"
        assert "obligation failed" in event.log_message()


class TestEntityCompensationEvent:
    def test_fields(self):
        event = EntityCompensationEvent(
            entity_type="load",
            entity_id="L-1",
            failed_state="ACCEPTED",
            reverted_to="NEGOTIATING",
            steps_compensated=2,
        )
        assert event.reverted_to == "NEGOTIATING"
        assert event.compensation_errors == []

    def test_with_errors(self):
        event = EntityCompensationEvent(
            entity_type="load",
            entity_id="L-1",
            failed_state="ACCEPTED",
            reverted_to="NEGOTIATING",
            steps_compensated=2,
            compensation_errors=["step2 failed"],
        )
        assert len(event.compensation_errors) == 1
        assert "compensation errors" in event.log_message()


class TestEventTaxonomy:
    def test_entity_events_tuple(self):
        from hexdag.kernel.orchestration.events import ENTITY_EVENTS

        assert StateTransitionEvent in ENTITY_EVENTS
        assert EntityGarbageCollected in ENTITY_EVENTS
        assert EntityObligationFailed in ENTITY_EVENTS
        assert EntityCompensationEvent in ENTITY_EVENTS
