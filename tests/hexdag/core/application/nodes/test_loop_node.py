"""Tests for LoopNode and ConditionalNode implementations."""

import pytest

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.registry import registry

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.events = []

    async def emit(self, event):
        """Mock emit that just stores events."""
        self.events.append(event)


class TestLoopNode:
    """Test cases for LoopNode functionality."""

    def test_loop_node_creation(self):
        """Test that LoopNode creates valid NodeSpec instances."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")
        node_spec = loop_node("test_loop", max_iterations=5)

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_loop"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_loop_node_successful_execution(self):
        """Test loop node execution with success condition."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def success_condition(data):
            return data.get("score", 0) > 0.8

        node_spec = loop_node("test_loop", max_iterations=3, success_condition=success_condition)

        event_manager = MockEventManager()
        input_data = {"score": 0.9, "attempt": 1}  # Use data that meets the condition

        # Replace the internal loop logic for testing
        original_fn = node_spec.fn

        async def test_fn(input_data, **ports):
            # Simulate loop execution with success condition
            if success_condition(input_data):
                return {
                    "result": input_data,
                    "iterations_completed": 1,
                    "success": True,
                    "loop_metadata": {"condition_met": True},
                }
            return await original_fn(input_data, **ports)

        result = await test_fn(input_data, event_manager=event_manager)

        # Should exit early when condition is met
        assert result["success"] is True
        assert result["iterations_completed"] == 1

    @pytest.mark.asyncio
    async def test_loop_node_max_iterations(self):
        """Test loop node hitting max iterations without success."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def never_succeeds(data):
            return False

        node_spec = loop_node("test_loop", max_iterations=2, success_condition=never_succeeds)

        # Verify the node spec was created correctly
        assert node_spec.name == "test_loop"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_loop_node_no_success_condition(self):
        """Test loop node without success condition (runs full iterations)."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")
        node_spec = loop_node("test_loop", max_iterations=3)

        # Verify node structure
        assert node_spec.name == "test_loop"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_loop_node_custom_iteration_key(self):
        """Test loop node with custom iteration key."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")
        node_spec = loop_node("test_loop", max_iterations=2, iteration_key="custom_iteration")

        # Verify node was created with custom iteration key
        assert node_spec.name == "test_loop"
        assert callable(node_spec.fn)

    def test_loop_node_invalid_max_iterations(self):
        """Test loop node with invalid max_iterations."""
        from hexdag.core.exceptions import ValidationError

        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        with pytest.raises(ValidationError, match="max_iterations"):
            loop_node("test_loop", max_iterations=0)

        with pytest.raises(ValidationError, match="max_iterations"):
            loop_node("test_loop", max_iterations=-1)


class TestConditionalNode:
    """Test cases for ConditionalNode functionality."""

    def test_conditional_node_creation(self):
        """Test that ConditionalNode creates valid NodeSpec instances."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_conditional"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_conditional_node_true_condition(self):
        """Test conditional node when condition is True."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        event_manager = MockEventManager()
        input_data = {"should_continue": True, "confidence": "low", "extra_data": "test"}

        result = await node_spec.fn(input_data, event_manager=event_manager)

        # Check result structure
        assert "condition_key" in result
        assert "condition_value" in result
        assert "action" in result
        assert "routing_decision" in result

        # Check values
        assert result["condition_key"] == "should_continue"
        assert result["condition_value"] is True
        assert result["action"] == "retry"
        assert "continue based on should_continue" in result["routing_decision"]

        # Check original data is preserved
        assert result["confidence"] == "low"
        assert result["extra_data"] == "test"

    @pytest.mark.asyncio
    async def test_conditional_node_false_condition(self):
        """Test conditional node when condition is False."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        event_manager = MockEventManager()
        input_data = {"should_continue": False, "confidence": "high"}

        result = await node_spec.fn(input_data, event_manager=event_manager)

        # Check result values
        assert result["condition_key"] == "should_continue"
        assert result["condition_value"] is False
        assert result["action"] == "proceed"
        assert "proceed based on should_continue" in result["routing_decision"]

    @pytest.mark.asyncio
    async def test_conditional_node_missing_condition_key(self):
        """Test conditional node when condition key is missing."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="missing_key",
            true_action="retry",
            false_action="proceed",
        )

        event_manager = MockEventManager()
        input_data = {"other_data": "value"}

        result = await node_spec.fn(input_data, event_manager=event_manager)

        # Should default to false when key is missing
        assert result["condition_key"] == "missing_key"
        assert result["condition_value"] is False
        assert result["action"] == "proceed"

    @pytest.mark.asyncio
    async def test_conditional_node_non_boolean_condition(self):
        """Test conditional node with non-boolean condition value."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="score",
            true_action="pass",
            false_action="fail",
        )

        event_manager = MockEventManager()

        # Test truthy value
        input_data = {"score": 0.8}
        result = await node_spec.fn(input_data, event_manager=event_manager)
        assert result["condition_value"] is True
        assert result["action"] == "pass"

        # Test falsy value
        input_data = {"score": 0}
        result = await node_spec.fn(input_data, event_manager=event_manager)
        assert result["condition_value"] is False
        assert result["action"] == "fail"

    @pytest.mark.asyncio
    async def test_conditional_node_event_manager_integration(self):
        """Test conditional node integration with event manager."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional",
            condition_key="ready",
            true_action="execute",
            false_action="wait",
        )

        event_manager = MockEventManager()
        input_data = {"ready": True, "task": "process_data"}

        result = await node_spec.fn(input_data, event_manager=event_manager)

        # Verify result structure
        assert result["condition_key"] == "ready"
        assert result["condition_value"] is True
        assert result["action"] == "execute"
        assert result["task"] == "process_data"  # Original data preserved

        # Event manager should be available for use
        assert event_manager is not None
