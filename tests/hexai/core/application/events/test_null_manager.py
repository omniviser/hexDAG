"""Tests for null manager implementations."""

import pytest

from hexai.core.application.events.context import ExecutionContext
from hexai.core.application.events.events import NodeStarted
from hexai.core.application.events.models import ControlSignal
from hexai.core.application.events.null_manager import (
    NULL_CONTROL_MANAGER,
    NULL_OBSERVER_MANAGER,
    NullControlManager,
    NullObserverManager,
    get_control_manager,
    get_observer_manager,
)


class TestNullControlManager:
    """Test NullControlManager for control plane."""

    @pytest.mark.asyncio
    async def test_always_returns_proceed(self):
        """Test that null control manager always returns PROCEED."""
        control_manager = NullControlManager()
        ctx = ExecutionContext(dag_id="test")

        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Should work with any event
        response = await control_manager.check("any event", "any context")
        assert response.signal == ControlSignal.PROCEED

    def test_register_is_noop(self):
        """Test that register does nothing."""
        control_manager = NullControlManager()

        # Should not raise
        control_manager.register("handler", priority=10, name="test")

    def test_clear_is_noop(self):
        """Test that clear does nothing."""
        control_manager = NullControlManager()

        # Should not raise
        control_manager.clear()


class TestNullObserverManager:
    """Test NullObserverManager for observability plane."""

    @pytest.mark.asyncio
    async def test_notify_is_silent(self):
        """Test that notify silently ignores events."""
        observer_manager = NullObserverManager()

        # Should not raise
        await observer_manager.notify(NodeStarted(name="test", wave_index=1))
        await observer_manager.notify("any event")

    def test_register_returns_dummy_id(self):
        """Test that register returns a dummy ID."""
        observer_manager = NullObserverManager()

        observer_id = observer_manager.register("observer")
        assert observer_id == "null-observer-id"

        # Custom ID is ignored
        observer_id = observer_manager.register("observer", observer_id="custom")
        assert observer_id == "null-observer-id"

    def test_unregister_returns_false(self):
        """Test that unregister always returns False."""
        observer_manager = NullObserverManager()

        result = observer_manager.unregister("any-id")
        assert result is False

    def test_always_empty(self):
        """Test that event dispatcher is always empty."""
        observer_manager = NullObserverManager()

        assert len(observer_manager) == 0

        # Even after "registering"
        observer_manager.register("observer")
        assert len(observer_manager) == 0


class TestGetterFunctions:
    """Test getter functions with fallback to null managers."""

    def test_get_event_bus_with_bus(self):
        """Test getting control manager when present."""
        from hexai.core.application.events.control_manager import ControlManager

        control_manager = ControlManager()
        ports = {"control_manager": control_manager}

        result = get_control_manager(ports)
        assert result is control_manager

    def test_get_event_bus_without_bus(self):
        """Test getting null control manager when not present."""
        ports = {}

        result = get_control_manager(ports)
        assert result is NULL_CONTROL_MANAGER

    def test_get_observer_manager_with_manager(self):
        """Test getting observer manager when present."""
        from hexai.core.application.events.observer_manager import ObserverManager

        observer_manager = ObserverManager()
        ports = {"observer_manager": observer_manager}

        result = get_observer_manager(ports)
        assert result is observer_manager

    def test_get_observer_manager_without_manager(self):
        """Test getting null observer manager when not present."""
        ports = {}

        result = get_observer_manager(ports)
        assert result is NULL_OBSERVER_MANAGER


class TestSingletonInstances:
    """Test that singleton instances work correctly."""

    @pytest.mark.asyncio
    async def test_singleton_bus(self):
        """Test singleton NULL_CONTROL_MANAGER."""
        ctx = ExecutionContext(dag_id="test")

        response = await NULL_CONTROL_MANAGER.check("event", ctx)
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_singleton_observer_manager(self):
        """Test singleton NULL_OBSERVER_MANAGER."""
        await NULL_OBSERVER_MANAGER.notify("event")
        assert len(NULL_OBSERVER_MANAGER) == 0

    @pytest.mark.asyncio
    async def test_can_use_together_in_orchestrator(self):
        """Test using NULL_CONTROL_MANAGER and NULL_OBSERVER_MANAGER together."""
        # This simulates how they would be used in orchestrator tests
        ctx = ExecutionContext(dag_id="test")
        event = NodeStarted(name="test", wave_index=1)

        # Control plane check
        response = await NULL_CONTROL_MANAGER.check(event, ctx)
        assert response.signal == ControlSignal.PROCEED

        # Observability notification
        await NULL_OBSERVER_MANAGER.notify(event)

        # Both work independently without interference
        assert len(NULL_OBSERVER_MANAGER) == 0
