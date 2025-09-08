"""Tests for null manager implementations."""

import pytest

from hexai.core.application.events.context import ExecutionContext
from hexai.core.application.events.events import NodeStarted
from hexai.core.application.events.models import ControlSignal
from hexai.core.application.events.null_manager import (
    NULL_EVENT_BUS,
    NULL_OBSERVER_MANAGER,
    NullEventBus,
    NullObserverManager,
    get_event_bus,
    get_observer_manager,
)


class TestNullEventBus:
    """Test NullEventBus for control plane."""

    @pytest.mark.asyncio
    async def test_always_returns_proceed(self):
        """Test that null bus always returns PROCEED."""
        bus = NullEventBus()
        ctx = ExecutionContext(dag_id="test")

        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Should work with any event
        response = await bus.check("any event", "any context")
        assert response.signal == ControlSignal.PROCEED

    def test_register_is_noop(self):
        """Test that register does nothing."""
        bus = NullEventBus()

        # Should not raise
        bus.register("handler", priority=10, name="test")

    def test_clear_is_noop(self):
        """Test that clear does nothing."""
        bus = NullEventBus()

        # Should not raise
        bus.clear()


class TestNullObserverManager:
    """Test NullObserverManager for observability plane."""

    @pytest.mark.asyncio
    async def test_notify_is_silent(self):
        """Test that notify silently ignores events."""
        manager = NullObserverManager()

        # Should not raise
        await manager.notify(NodeStarted(name="test", wave_index=1))
        await manager.notify("any event")

    def test_register_returns_dummy_id(self):
        """Test that register returns a dummy ID."""
        manager = NullObserverManager()

        observer_id = manager.register("observer")
        assert observer_id == "null-observer-id"

        # Custom ID is ignored
        observer_id = manager.register("observer", observer_id="custom")
        assert observer_id == "null-observer-id"

    def test_unregister_returns_false(self):
        """Test that unregister always returns False."""
        manager = NullObserverManager()

        result = manager.unregister("any-id")
        assert result is False

    def test_always_empty(self):
        """Test that manager is always empty."""
        manager = NullObserverManager()

        assert len(manager) == 0

        # Even after "registering"
        manager.register("observer")
        assert len(manager) == 0


class TestGetterFunctions:
    """Test getter functions with fallback to null managers."""

    def test_get_event_bus_with_bus(self):
        """Test getting event bus when present."""
        from hexai.core.application.events.bus import EventBus

        bus = EventBus()
        ports = {"event_bus": bus}

        result = get_event_bus(ports)
        assert result is bus

    def test_get_event_bus_without_bus(self):
        """Test getting null bus when not present."""
        ports = {}

        result = get_event_bus(ports)
        assert result is NULL_EVENT_BUS

    def test_get_observer_manager_with_manager(self):
        """Test getting observer manager when present."""
        from hexai.core.application.events.manager import ObserverManager

        manager = ObserverManager()
        ports = {"observer_manager": manager}

        result = get_observer_manager(ports)
        assert result is manager

    def test_get_observer_manager_without_manager(self):
        """Test getting null manager when not present."""
        ports = {}

        result = get_observer_manager(ports)
        assert result is NULL_OBSERVER_MANAGER


class TestSingletonInstances:
    """Test that singleton instances work correctly."""

    @pytest.mark.asyncio
    async def test_singleton_bus(self):
        """Test singleton NULL_EVENT_BUS."""
        ctx = ExecutionContext(dag_id="test")

        response = await NULL_EVENT_BUS.check("event", ctx)
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_singleton_observer_manager(self):
        """Test singleton NULL_OBSERVER_MANAGER."""
        await NULL_OBSERVER_MANAGER.notify("event")
        assert len(NULL_OBSERVER_MANAGER) == 0

    @pytest.mark.asyncio
    async def test_can_use_together_in_orchestrator(self):
        """Test using NULL_EVENT_BUS and NULL_OBSERVER_MANAGER together."""
        # This simulates how they would be used in orchestrator tests
        ctx = ExecutionContext(dag_id="test")
        event = NodeStarted(name="test", wave_index=1)

        # Control plane check
        response = await NULL_EVENT_BUS.check(event, ctx)
        assert response.signal == ControlSignal.PROCEED

        # Observability notification
        await NULL_OBSERVER_MANAGER.notify(event)

        # Both work independently without interference
        assert len(NULL_OBSERVER_MANAGER) == 0
