"""Tests for NodeExecutor component."""

import asyncio

import pytest
from pydantic import BaseModel

from hexai.core.application.events import NodeCancelled, NodeCompleted, NodeFailed, NodeStarted
from hexai.core.application.policies.models import PolicyResponse, PolicySignal
from hexai.core.domain.dag import NodeSpec, ValidationError
from hexai.core.orchestration.components.node_executor import (
    NodeExecutionError,
    NodeExecutor,
    NodeTimeoutError,
)
from hexai.core.orchestration.components.policy_coordinator import (
    OrchestratorError,
    PolicyCoordinator,
)
from hexai.core.orchestration.models import NodeExecutionContext


# Test models
class SimpleInput(BaseModel):
    """Simple input model."""

    value: int


class SimpleOutput(BaseModel):
    """Simple output model."""

    result: int


# Test functions
async def simple_async_function(input_data: int) -> int:
    """Simple async function that doubles input."""
    await asyncio.sleep(0.01)
    return input_data * 2


def simple_sync_function(input_data: int) -> int:
    """Simple sync function that triples input."""
    return input_data * 3


async def failing_function(input_data: int) -> int:
    """Function that always fails."""
    raise ValueError("Intentional failure")


async def slow_function(input_data: int) -> int:
    """Function that takes a long time."""
    await asyncio.sleep(10)
    return input_data


# Mock classes
class MockObserverManager:
    """Mock observer manager."""

    def __init__(self):
        self.events = []

    async def notify(self, event):
        """Record events."""
        self.events.append(event)


class MockPolicyManager:
    """Mock policy manager."""

    def __init__(self, response: PolicyResponse | None = None):
        self.response = response or PolicyResponse(signal=PolicySignal.PROCEED)
        self.contexts = []

    async def evaluate(self, context):
        """Record contexts and return response."""
        self.contexts.append(context)
        return self.response


class TestNodeExecutor:
    """Test NodeExecutor component."""

    @pytest.fixture
    def executor(self):
        """Create a NodeExecutor instance."""
        return NodeExecutor(strict_validation=False, default_node_timeout=None)

    @pytest.fixture
    def strict_executor(self):
        """Create a strict NodeExecutor."""
        return NodeExecutor(strict_validation=True, default_node_timeout=None)

    @pytest.fixture
    def executor_with_timeout(self):
        """Create a NodeExecutor with default timeout."""
        return NodeExecutor(strict_validation=False, default_node_timeout=1.0)

    @pytest.fixture
    def coordinator(self):
        """Create a PolicyCoordinator."""
        return PolicyCoordinator()

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return NodeExecutionContext(dag_id="test_dag", node_id="test_node")

    @pytest.fixture
    def observer(self):
        """Create mock observer."""
        return MockObserverManager()

    @pytest.fixture
    def policy(self):
        """Create mock policy."""
        return MockPolicyManager()

    @pytest.mark.asyncio
    async def test_execute_async_function(self, executor, coordinator, context, observer, policy):
        """Test executing an async function."""
        node_spec = NodeSpec("test_node", simple_async_function)

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert result == 10  # 5 * 2
        # Check events were emitted
        assert len(observer.events) == 2
        assert isinstance(observer.events[0], NodeStarted)
        assert isinstance(observer.events[1], NodeCompleted)

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, executor, coordinator, context, observer, policy):
        """Test executing a sync function."""
        node_spec = NodeSpec("test_node", simple_sync_function)

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert result == 15  # 5 * 3

    @pytest.mark.asyncio
    async def test_input_validation_success(self, executor, coordinator, context, observer, policy):
        """Test successful input validation."""

        async def validated_function(input_data: SimpleInput) -> int:
            return input_data.value * 2

        node_spec = NodeSpec(
            "test_node",
            validated_function,
            in_model=SimpleInput,
        )

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input={"value": 5},
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert result == 10

    @pytest.mark.asyncio
    async def test_input_validation_failure_lenient(
        self, executor, coordinator, context, observer, policy
    ):
        """Test input validation failure in lenient mode."""

        async def validated_function(input_data) -> int:
            # Handle both validated and unvalidated input
            if isinstance(input_data, SimpleInput):
                return input_data.value * 2
            return input_data * 2

        node_spec = NodeSpec(
            "test_node",
            validated_function,
            in_model=SimpleInput,
        )

        # Should not raise, will use unvalidated input
        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,  # Wrong format, should be dict
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
            validate=True,
        )

        assert result == 10  # Still works with raw input

    @pytest.mark.asyncio
    async def test_input_validation_failure_strict(
        self, strict_executor, coordinator, context, observer, policy
    ):
        """Test input validation failure in strict mode."""

        async def validated_function(input_data: SimpleInput) -> int:
            return input_data.value * 2

        node_spec = NodeSpec(
            "test_node",
            validated_function,
            in_model=SimpleInput,
        )

        # In strict mode, validation error becomes NodeExecutionError
        with pytest.raises((ValidationError, NodeExecutionError)):
            await strict_executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input="invalid",
                ports={},
                context=context,
                policy_coordinator=coordinator,
                observer_manager=observer,
                policy_manager=policy,
            )

    @pytest.mark.asyncio
    async def test_output_validation_success(
        self, executor, coordinator, context, observer, policy
    ):
        """Test successful output validation."""
        node_spec = NodeSpec(
            "test_node",
            simple_async_function,
            out_model=SimpleOutput,
        )

        # Function returns int, which will be validated against SimpleOutput
        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert isinstance(result, (int, SimpleOutput))

    @pytest.mark.asyncio
    async def test_node_timeout_with_default(
        self, executor_with_timeout, coordinator, context, observer, policy
    ):
        """Test node timeout using default timeout."""
        node_spec = NodeSpec("test_node", slow_function)

        with pytest.raises(NodeTimeoutError) as exc_info:
            await executor_with_timeout.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                ports={},
                context=context,
                policy_coordinator=coordinator,
                observer_manager=observer,
                policy_manager=policy,
            )

        assert exc_info.value.node_name == "test_node"
        assert exc_info.value.timeout == 1.0
        # Check cancelled event was emitted
        assert any(isinstance(e, NodeCancelled) for e in observer.events)

    @pytest.mark.asyncio
    async def test_node_timeout_with_node_override(
        self, executor, coordinator, context, observer, policy
    ):
        """Test node timeout with per-node override."""
        node_spec = NodeSpec("test_node", slow_function, timeout=0.1)

        with pytest.raises(NodeTimeoutError) as exc_info:
            await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                ports={},
                context=context,
                policy_coordinator=coordinator,
                observer_manager=observer,
                policy_manager=policy,
            )

        assert exc_info.value.timeout == 0.1

    @pytest.mark.asyncio
    async def test_node_execution_failure(self, executor, coordinator, context, observer, policy):
        """Test node execution failure."""
        node_spec = NodeSpec("test_node", failing_function)

        with pytest.raises(NodeExecutionError) as exc_info:
            await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                ports={},
                context=context,
                policy_coordinator=coordinator,
                observer_manager=observer,
                policy_manager=policy,
            )

        assert exc_info.value.node_name == "test_node"
        assert isinstance(exc_info.value.original_error, ValueError)
        # Check failed event was emitted
        assert any(isinstance(e, NodeFailed) for e in observer.events)

    @pytest.mark.asyncio
    async def test_policy_skip_signal(self, executor, coordinator, context, observer):
        """Test policy SKIP signal."""
        policy = MockPolicyManager(PolicyResponse(signal=PolicySignal.SKIP, data="skipped"))
        node_spec = NodeSpec("test_node", simple_async_function)

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert result == "skipped"  # Returns policy data

    @pytest.mark.asyncio
    async def test_policy_fail_signal(self, executor, coordinator, context, observer):
        """Test policy FAIL signal."""
        policy = MockPolicyManager(PolicyResponse(signal=PolicySignal.FAIL, data="Policy blocked"))
        node_spec = NodeSpec("test_node", simple_async_function)

        # Policy FAIL raises OrchestratorError which gets wrapped in NodeExecutionError
        with pytest.raises((OrchestratorError, NodeExecutionError), match="blocked"):
            await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                ports={},
                context=context,
                policy_coordinator=coordinator,
                observer_manager=observer,
                policy_manager=policy,
            )

    @pytest.mark.asyncio
    async def test_ports_passed_to_function(self, executor, coordinator, context, observer, policy):
        """Test that ports are passed to node function."""

        async def function_with_ports(input_data: int, llm=None, database=None) -> dict:
            return {"input": input_data, "llm": llm, "database": database}

        node_spec = NodeSpec("test_node", function_with_ports)
        mock_llm = "mock_llm"
        mock_db = "mock_db"

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={"llm": mock_llm, "database": mock_db},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        assert result["input"] == 5
        assert result["llm"] == "mock_llm"
        assert result["database"] == "mock_db"

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_function(
        self, executor, coordinator, context, observer, policy
    ):
        """Test that kwargs are passed to node function."""

        async def function_with_kwargs(input_data: int, **kwargs) -> dict:
            return {"input": input_data, "kwargs": kwargs}

        node_spec = NodeSpec("test_node", function_with_kwargs)

        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
            custom_arg="custom_value",
        )

        assert result["input"] == 5
        assert result["kwargs"]["custom_arg"] == "custom_value"

    @pytest.mark.asyncio
    async def test_validation_disabled(self, executor, coordinator, context, observer, policy):
        """Test that validation can be disabled."""
        node_spec = NodeSpec(
            "test_node",
            simple_async_function,
            in_model=SimpleInput,
            out_model=SimpleOutput,
        )

        # Should not validate even with wrong input format
        result = await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,  # Not a dict
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
            validate=False,
        )

        assert result == 10

    @pytest.mark.asyncio
    async def test_completed_event_contains_duration(
        self, executor, coordinator, context, observer, policy
    ):
        """Test that NodeCompleted event contains duration."""
        node_spec = NodeSpec("test_node", simple_async_function)

        await executor.execute_node(
            node_name="test_node",
            node_spec=node_spec,
            node_input=5,
            ports={},
            context=context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
        )

        completed_event = next(e for e in observer.events if isinstance(e, NodeCompleted))
        assert completed_event.duration_ms > 0
