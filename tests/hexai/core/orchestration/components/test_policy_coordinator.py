"""Tests for PolicyCoordinator component."""

import pytest

from hexai.core.application.events import NodeStarted
from hexai.core.application.policies.models import PolicyContext, PolicyResponse, PolicySignal
from hexai.core.orchestration.components.policy_coordinator import (
    OrchestratorError,
    PolicyCoordinator,
)
from hexai.core.orchestration.models import NodeExecutionContext


class MockObserverManager:
    """Mock observer manager for testing."""

    def __init__(self):
        self.events = []

    async def notify(self, event):
        """Record notified events."""
        self.events.append(event)


class MockPolicyManager:
    """Mock policy manager for testing."""

    def __init__(self, response: PolicyResponse | None = None):
        self.response = response or PolicyResponse()
        self.contexts = []

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Record evaluation contexts and return response."""
        self.contexts.append(context)
        return self.response


class TestPolicyCoordinator:
    """Test PolicyCoordinator component."""

    @pytest.fixture
    def coordinator(self):
        """Create a PolicyCoordinator instance."""
        return PolicyCoordinator()

    @pytest.fixture
    def execution_context(self):
        """Create a NodeExecutionContext."""
        return NodeExecutionContext(dag_id="test_dag", node_id="test_node", wave_index=0)

    @pytest.mark.asyncio
    async def test_notify_observer_with_observer(self, coordinator):
        """Test notifying observer when observer exists."""
        observer = MockObserverManager()
        event = NodeStarted(name="test_node", wave_index=0)

        await coordinator.notify_observer(observer, event)

        assert len(observer.events) == 1
        assert observer.events[0] is event

    @pytest.mark.asyncio
    async def test_notify_observer_without_observer(self, coordinator):
        """Test notifying observer when observer is None."""
        event = NodeStarted(name="test_node", wave_index=0)

        # Should not raise an error
        await coordinator.notify_observer(None, event)

    @pytest.mark.asyncio
    async def test_evaluate_policy_with_policy_manager(self, coordinator, execution_context):
        """Test evaluating policy with a policy manager."""
        policy_manager = MockPolicyManager(PolicyResponse(signal=PolicySignal.SKIP))
        event = NodeStarted(name="test_node", wave_index=0)

        response = await coordinator.evaluate_policy(
            policy_manager, event, execution_context, node_id="custom_node", wave_index=1
        )

        assert response.signal == PolicySignal.SKIP
        assert len(policy_manager.contexts) == 1
        context = policy_manager.contexts[0]
        assert context.dag_id == "test_dag"
        assert context.node_id == "custom_node"  # Override worked
        assert context.wave_index == 1  # Override worked

    @pytest.mark.asyncio
    async def test_evaluate_policy_without_policy_manager(self, coordinator, execution_context):
        """Test evaluating policy without a policy manager."""
        event = NodeStarted(name="test_node", wave_index=0)

        response = await coordinator.evaluate_policy(None, event, execution_context)

        # Should return default PROCEED response
        assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_evaluate_policy_uses_context_defaults(self, coordinator, execution_context):
        """Test that evaluate_policy uses context values when overrides not provided."""
        policy_manager = MockPolicyManager()
        event = NodeStarted(name="test_node", wave_index=0)

        await coordinator.evaluate_policy(policy_manager, event, execution_context)

        context = policy_manager.contexts[0]
        assert context.node_id == "test_node"  # From execution_context
        assert context.wave_index == 0  # From execution_context
        assert context.attempt == 1  # Default

    @pytest.mark.asyncio
    async def test_evaluate_policy_with_attempt(self, coordinator, execution_context):
        """Test evaluating policy with custom attempt number."""
        policy_manager = MockPolicyManager()
        event = NodeStarted(name="test_node", wave_index=0)

        await coordinator.evaluate_policy(policy_manager, event, execution_context, attempt=3)

        context = policy_manager.contexts[0]
        assert context.attempt == 3

    def test_check_policy_signal_proceed(self, coordinator):
        """Test check_policy_signal with PROCEED signal."""
        response = PolicyResponse(signal=PolicySignal.PROCEED)

        # Should not raise
        coordinator.check_policy_signal(response, "Test context")

    def test_check_policy_signal_skip(self, coordinator):
        """Test check_policy_signal with SKIP signal."""
        response = PolicyResponse(signal=PolicySignal.SKIP)

        with pytest.raises(OrchestratorError, match="Test context blocked: skip"):
            coordinator.check_policy_signal(response, "Test context")

    def test_check_policy_signal_fail(self, coordinator):
        """Test check_policy_signal with FAIL signal."""
        response = PolicyResponse(signal=PolicySignal.FAIL)

        with pytest.raises(OrchestratorError, match="Test context blocked: fail"):
            coordinator.check_policy_signal(response, "Test context")

    def test_check_policy_signal_retry(self, coordinator):
        """Test check_policy_signal with RETRY signal."""
        response = PolicyResponse(signal=PolicySignal.RETRY)

        with pytest.raises(OrchestratorError, match="Test context blocked: retry"):
            coordinator.check_policy_signal(response, "Test context")

    @pytest.mark.asyncio
    async def test_integration_notify_and_evaluate(self, coordinator, execution_context):
        """Test integration of notify and evaluate."""
        observer = MockObserverManager()
        policy_manager = MockPolicyManager(PolicyResponse(signal=PolicySignal.PROCEED))
        event = NodeStarted(name="test_node", wave_index=0)

        # Notify observer
        await coordinator.notify_observer(observer, event)

        # Evaluate policy
        response = await coordinator.evaluate_policy(policy_manager, event, execution_context)

        # Both should have recorded the event/context
        assert len(observer.events) == 1
        assert len(policy_manager.contexts) == 1
        assert response.signal == PolicySignal.PROCEED
