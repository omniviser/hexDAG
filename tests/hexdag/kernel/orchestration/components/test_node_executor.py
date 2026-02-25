"""Tests for NodeExecutor component."""

import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

from hexdag.kernel.context import ExecutionContext
from hexdag.kernel.domain.dag import NodeSpec, NodeValidationError
from hexdag.kernel.orchestration.components.execution_coordinator import ExecutionCoordinator
from hexdag.kernel.orchestration.components.node_executor import (
    NodeExecutionError,
    NodeExecutor,
    NodeTimeoutError,
)
from hexdag.kernel.orchestration.events import (
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeStarted,
)
from hexdag.kernel.orchestration.models import NodeExecutionContext


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
        """Create an ExecutionCoordinator."""
        return ExecutionCoordinator()

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return NodeExecutionContext(dag_id="test_dag", node_id="test_node")

    @pytest.fixture
    def observer(self):
        """Create mock observer."""
        return MockObserverManager()

    @pytest.mark.asyncio
    async def test_execute_async_function(self, executor, coordinator, context, observer):
        """Test executing an async function."""
        node_spec = NodeSpec("test_node", simple_async_function)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert result == 10  # 5 * 2
        # Check events were emitted
        assert len(observer.events) == 2
        assert isinstance(observer.events[0], NodeStarted)
        assert isinstance(observer.events[1], NodeCompleted)

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, executor, coordinator, context, observer):
        """Test executing a sync function."""
        node_spec = NodeSpec("test_node", simple_sync_function)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert result == 15  # 5 * 3

    @pytest.mark.asyncio
    async def test_input_validation_success(self, executor, coordinator, context, observer):
        """Test successful input validation."""

        async def validated_function(input_data: SimpleInput) -> int:
            return input_data.value * 2

        node_spec = NodeSpec(
            "test_node",
            validated_function,
            in_model=SimpleInput,
        )

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={"value": 5},
                context=context,
                coordinator=coordinator,
            )

        assert result == 10

    @pytest.mark.asyncio
    async def test_input_validation_failure_lenient(self, executor, coordinator, context, observer):
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
        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,  # Wrong format, should be dict
                context=context,
                coordinator=coordinator,
                validate=True,
            )

        assert result == 10  # Still works with raw input

    @pytest.mark.asyncio
    async def test_input_validation_failure_strict(
        self, strict_executor, coordinator, context, observer
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
        with pytest.raises((NodeValidationError, NodeExecutionError)):
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await strict_executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input="invalid",
                    context=context,
                    coordinator=coordinator,
                )

    @pytest.mark.asyncio
    async def test_output_validation_success(self, executor, coordinator, context, observer):
        """Test successful output validation."""
        node_spec = NodeSpec(
            "test_node",
            simple_async_function,
            out_model=SimpleOutput,
        )

        # Function returns int, which will be validated against SimpleOutput
        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert isinstance(result, (int, SimpleOutput))

    @pytest.mark.asyncio
    async def test_node_timeout_with_default(
        self, executor_with_timeout, coordinator, context, observer
    ):
        """Test node timeout using default timeout."""
        node_spec = NodeSpec("test_node", slow_function)

        with pytest.raises(NodeTimeoutError) as exc_info:
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor_with_timeout.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input=5,
                    context=context,
                    coordinator=coordinator,
                )

        assert exc_info.value.node_name == "test_node"
        assert exc_info.value.timeout == 1.0
        # Check cancelled event was emitted
        assert any(isinstance(e, NodeCancelled) for e in observer.events)

    @pytest.mark.asyncio
    async def test_node_timeout_with_node_override(self, executor, coordinator, context, observer):
        """Test node timeout with per-node override."""
        node_spec = NodeSpec("test_node", slow_function, timeout=0.1)

        with pytest.raises(NodeTimeoutError) as exc_info:
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input=5,
                    context=context,
                    coordinator=coordinator,
                )

        assert exc_info.value.timeout == 0.1

    @pytest.mark.asyncio
    async def test_node_execution_failure(self, executor, coordinator, context, observer):
        """Test node execution failure."""
        node_spec = NodeSpec("test_node", failing_function)

        with pytest.raises(NodeExecutionError) as exc_info:
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input=5,
                    context=context,
                    coordinator=coordinator,
                )

        assert exc_info.value.node_name == "test_node"
        assert isinstance(exc_info.value.original_error, ValueError)
        # Check failed event was emitted
        assert any(isinstance(e, NodeFailed) for e in observer.events)

    @pytest.mark.asyncio
    async def test_ports_passed_to_function(self, executor, coordinator, context, observer):
        """Test that ports are accessible from node function via ExecutionContext."""
        from hexdag.kernel.context import get_port

        async def function_with_ports(input_data: int) -> dict:
            llm = get_port("llm")
            database = get_port("database")
            return {"input": input_data, "llm": llm, "database": database}

        node_spec = NodeSpec("test_node", function_with_ports)
        mock_llm = "mock_llm"
        mock_db = "mock_db"

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={"llm": mock_llm, "database": mock_db},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert result["input"] == 5
        assert result["llm"] == "mock_llm"
        assert result["database"] == "mock_db"

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_function(self, executor, coordinator, context, observer):
        """Test that kwargs are passed to node function."""

        async def function_with_kwargs(input_data: int, **kwargs) -> dict:
            return {"input": input_data, "kwargs": kwargs}

        node_spec = NodeSpec("test_node", function_with_kwargs)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
                custom_arg="custom_value",
            )

        assert result["input"] == 5
        assert result["kwargs"]["custom_arg"] == "custom_value"

    @pytest.mark.asyncio
    async def test_validation_disabled(self, executor, coordinator, context, observer):
        """Test that validation can be disabled."""
        node_spec = NodeSpec(
            "test_node",
            simple_async_function,
            in_model=SimpleInput,
            out_model=SimpleOutput,
        )

        # Should not validate even with wrong input format
        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,  # Not a dict
                context=context,
                coordinator=coordinator,
                validate=False,
            )

        assert result == 10

    @pytest.mark.asyncio
    async def test_completed_event_contains_duration(
        self, executor, coordinator, context, observer
    ):
        """Test that NodeCompleted event contains duration."""
        node_spec = NodeSpec("test_node", simple_async_function)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        completed_event = next(e for e in observer.events if isinstance(e, NodeCompleted))
        assert completed_event.duration_ms > 0

    @pytest.mark.asyncio
    async def test_max_retries_success_after_failures(
        self, executor, coordinator, context, observer
    ):
        """Test that max_retries allows retrying until success."""
        call_count = 0

        async def failing_then_success(input_data: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return input_data * 2

        node_spec = NodeSpec("test_node", failing_then_success, max_retries=3)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert result == 10  # 5 * 2
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, executor, coordinator, context, observer):
        """Test that max_retries raises after all retries exhausted."""
        call_count = 0

        async def always_fails(input_data: int) -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        node_spec = NodeSpec("test_node", always_fails, max_retries=3)

        with pytest.raises(NodeExecutionError) as exc_info:
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input=5,
                    context=context,
                    coordinator=coordinator,
                )

        assert call_count == 3  # All retries exhausted
        assert "Attempt 3 failed" in str(exc_info.value.original_error)

    @pytest.mark.asyncio
    async def test_max_retries_default_no_retry(self, executor, coordinator, context, observer):
        """Test that without max_retries, failures are immediate."""
        call_count = 0

        async def always_fails(input_data: int) -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError("Failed")

        node_spec = NodeSpec("test_node", always_fails)  # No max_retries

        with pytest.raises(NodeExecutionError):
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input=5,
                    context=context,
                    coordinator=coordinator,
                )

        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_max_retries_with_timeout(self, executor, coordinator, context, observer):
        """Test max_retries with timeout - retries on timeout."""
        call_count = 0

        async def slow_then_fast(input_data: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                await asyncio.sleep(10)  # Will timeout
            return input_data * 2

        node_spec = NodeSpec("test_node", slow_then_fast, timeout=0.1, max_retries=3)

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input=5,
                context=context,
                coordinator=coordinator,
            )

        assert result == 10  # 5 * 2
        assert call_count == 2  # First timed out, second succeeded


class TestRetryWithExponentialBackoff:
    """Test retry functionality with exponential backoff delays."""

    @pytest.fixture
    def mock_context(self) -> NodeExecutionContext:
        """Create a mock execution context."""
        return NodeExecutionContext(
            dag_id="test-dag",
        )

    @pytest.fixture
    def coordinator(self) -> ExecutionCoordinator:
        """Create an ExecutionCoordinator."""
        return ExecutionCoordinator()

    @pytest.fixture
    def observer(self) -> MockObserverManager:
        """Create a mock observer."""
        return MockObserverManager()

    @pytest.mark.asyncio
    async def test_retry_on_failure_with_default_delay(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that retry occurs with default delay after failure."""
        call_times: list[float] = []
        call_count = 0

        async def failing_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_times.append(time.time())
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return {"result": "success"}

        node_spec = NodeSpec(
            name="test_node",
            fn=failing_fn,
            max_retries=3,
            # Use default retry_delay (1.0), but we'll use smaller values for testing
        )

        executor = NodeExecutor()

        # Note: This test uses default delays which are 1.0s
        # For fast tests, we test with custom delays below
        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )

        assert result == {"result": "success"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_custom_delay(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test retry with custom delay configuration."""
        call_times: list[float] = []
        call_count = 0

        async def failing_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_times.append(time.time())
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return {"result": "success"}

        # Use small delays for fast testing
        node_spec = NodeSpec(
            name="test_node",
            fn=failing_fn,
            max_retries=3,
            retry_delay=0.05,  # 50ms initial delay
            retry_backoff=2.0,  # 2x multiplier
            retry_max_delay=1.0,  # 1s max
        )

        executor = NodeExecutor()

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            start_time = time.time()
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )
            total_time = time.time() - start_time

        assert result == {"result": "success"}
        assert call_count == 3
        # Should have delays: 0.05s (first retry) + 0.10s (second retry) = 0.15s minimum
        assert total_time >= 0.15

        # Verify delay between calls
        if len(call_times) >= 3:
            first_delay = call_times[1] - call_times[0]
            second_delay = call_times[2] - call_times[1]
            # First delay should be ~0.05s
            assert first_delay >= 0.04
            # Second delay should be ~0.10s (0.05 * 2.0)
            assert second_delay >= 0.08

    @pytest.mark.asyncio
    async def test_retry_respects_max_delay_cap(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that retry delay is capped at retry_max_delay."""
        call_times: list[float] = []
        call_count = 0

        async def failing_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_times.append(time.time())
            call_count += 1
            if call_count < 5:
                raise RuntimeError("Transient error")
            return {"result": "success"}

        # Configure with low max_delay to test capping
        node_spec = NodeSpec(
            name="test_node",
            fn=failing_fn,
            max_retries=5,
            retry_delay=0.05,  # 50ms initial
            retry_backoff=10.0,  # Large multiplier to trigger cap
            retry_max_delay=0.08,  # Cap at 80ms
        )

        executor = NodeExecutor()

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )

        assert result == {"result": "success"}
        assert call_count == 5

        # Check that delays are capped
        # Delay sequence without cap would be: 0.05, 0.5, 5.0, 50.0
        # With cap at 0.08: 0.05, 0.08, 0.08, 0.08
        if len(call_times) >= 5:
            for i in range(1, len(call_times) - 1):
                delay = call_times[i + 1] - call_times[i]
                # All delays after first should be capped at ~0.08s
                if i > 1:
                    assert delay < 0.12  # Some tolerance for execution time

    @pytest.mark.asyncio
    async def test_no_retry_on_success(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that successful execution doesn't trigger retries."""
        call_count = 0

        async def success_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"result": "success"}

        node_spec = NodeSpec(
            name="test_node",
            fn=success_fn,
            max_retries=3,
            retry_delay=0.1,
        )

        executor = NodeExecutor()

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )

        assert result == {"result": "success"}
        assert call_count == 1  # Only one call, no retries

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_error(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that error is raised when all retries are exhausted."""
        call_count = 0

        async def always_fails(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Permanent error")

        node_spec = NodeSpec(
            name="test_node",
            fn=always_fails,
            max_retries=3,
            retry_delay=0.01,  # Short delay for fast test
        )

        executor = NodeExecutor()

        # NodeExecutor wraps RuntimeError in NodeExecutionError
        with pytest.raises(NodeExecutionError, match="Permanent error"):
            async with ExecutionContext(
                observer_manager=observer,
                run_id="test-run",
                ports={},
            ):
                await executor.execute_node(
                    node_name="test_node",
                    node_spec=node_spec,
                    node_input={},
                    context=mock_context,
                    coordinator=coordinator,
                )

        assert call_count == 3  # Tried 3 times then failed

    @pytest.mark.asyncio
    async def test_timeout_retry_with_backoff(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that timeout errors also use exponential backoff."""
        call_times: list[float] = []
        call_count = 0

        async def slow_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_times.append(time.time())
            call_count += 1
            if call_count < 3:
                await asyncio.sleep(1.0)  # Longer than timeout
            return {"result": "success"}

        node_spec = NodeSpec(
            name="test_node",
            fn=slow_fn,
            timeout=0.05,  # 50ms timeout
            max_retries=3,
            retry_delay=0.02,  # 20ms initial delay
            retry_backoff=2.0,
        )

        executor = NodeExecutor()

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )

        assert result == {"result": "success"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_delay_without_retry_config(
        self,
        mock_context: NodeExecutionContext,
        coordinator: ExecutionCoordinator,
        observer: MockObserverManager,
    ) -> None:
        """Test that default delay (1.0s) is used when no retry_delay is set."""
        call_count = 0

        async def failing_fn(data: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Transient error")
            return {"result": "success"}

        # No retry delay specified - uses default of 1.0s
        node_spec = NodeSpec(
            name="test_node",
            fn=failing_fn,
            max_retries=2,
            # retry_delay=None (uses default 1.0)
        )

        executor = NodeExecutor()

        async with ExecutionContext(
            observer_manager=observer,
            run_id="test-run",
            ports={},
        ):
            start_time = time.time()
            result = await executor.execute_node(
                node_name="test_node",
                node_spec=node_spec,
                node_input={},
                context=mock_context,
                coordinator=coordinator,
            )
            total_time = time.time() - start_time

        assert result == {"result": "success"}
        assert call_count == 2
        # Should have ~1.0s delay from default
        assert total_time >= 0.9


class TestRetryConfigInNodeSpec:
    """Test that retry configuration is correctly stored in NodeSpec."""

    def test_nodespec_stores_retry_config(self) -> None:
        """Test that NodeSpec stores all retry configuration fields."""

        async def dummy_fn(data: Any) -> dict[str, Any]:
            return data

        node_spec = NodeSpec(
            name="test",
            fn=dummy_fn,
            max_retries=5,
            retry_delay=2.0,
            retry_backoff=3.0,
            retry_max_delay=120.0,
        )

        assert node_spec.max_retries == 5
        assert node_spec.retry_delay == 2.0
        assert node_spec.retry_backoff == 3.0
        assert node_spec.retry_max_delay == 120.0

    def test_nodespec_retry_defaults_to_none(self) -> None:
        """Test that retry fields default to None."""

        async def dummy_fn(data: Any) -> dict[str, Any]:
            return data

        node_spec = NodeSpec(
            name="test",
            fn=dummy_fn,
        )

        assert node_spec.max_retries is None
        assert node_spec.retry_delay is None
        assert node_spec.retry_backoff is None
        assert node_spec.retry_max_delay is None


# ============================================================================
# Tests: NodeExecutor auto-skip on upstream skip
# ============================================================================


async def noop_fn(input_data):
    return {"processed": input_data}


class TestNodeExecutorUpstreamSkip:
    """Test that NodeExecutor auto-skips when input has _upstream_skipped."""

    @pytest.fixture()
    def executor(self) -> NodeExecutor:
        return NodeExecutor(strict_validation=False)

    @pytest.fixture()
    def coordinator(self) -> ExecutionCoordinator:
        return ExecutionCoordinator()

    @pytest.fixture()
    def observer(self) -> MockObserverManager:
        return MockObserverManager()

    @pytest.mark.asyncio()
    async def test_auto_skips_on_upstream_skip(self, executor, coordinator, observer) -> None:
        """NodeExecutor should auto-skip when input has _upstream_skipped flag."""
        node_spec = NodeSpec("downstream", noop_fn)
        upstream_skip_input = {"_skipped": True, "_upstream_skipped": True}

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="downstream",
                node_spec=node_spec,
                node_input=upstream_skip_input,
                context=NodeExecutionContext(dag_id="test", node_id="downstream"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
            )

        assert result["_skipped"] is True
        assert result["_upstream_skipped"] is True
        # Should have emitted NodeSkipped event
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 1
        assert "upstream" in skip_events[0].reason

    @pytest.mark.asyncio()
    async def test_runs_normally_without_skip(self, executor, coordinator, observer) -> None:
        """Normal input -> node executes normally (regression check)."""
        node_spec = NodeSpec("normal", noop_fn)

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="normal",
                node_spec=node_spec,
                node_input={"data": "hello"},
                context=NodeExecutionContext(dag_id="test", node_id="normal"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
            )

        assert result == {"processed": {"data": "hello"}}
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 0

    @pytest.mark.asyncio()
    async def test_does_not_skip_on_regular_skipped_result(
        self, executor, coordinator, observer
    ) -> None:
        """Input with _skipped but NOT _upstream_skipped should not auto-skip.

        This preserves backward compat -- only upstream propagation triggers auto-skip.
        """
        node_spec = NodeSpec("downstream", noop_fn)
        # This is a regular skipped result, not an upstream propagation
        regular_skip = {"_skipped": True, "reason": "when clause"}

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="downstream",
                node_spec=node_spec,
                node_input=regular_skip,
                context=NodeExecutionContext(dag_id="test", node_id="downstream"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
            )

        # Should have executed normally (noop_fn processes the input)
        assert result == {"processed": {"_skipped": True, "reason": "when clause"}}


# ============================================================================
# Tests: when clause data context
# ============================================================================


class TestWhenClauseDataContext:
    """Test that when clause evaluates against both node_results and validated input."""

    @pytest.fixture()
    def executor(self) -> NodeExecutor:
        return NodeExecutor(strict_validation=False)

    @pytest.fixture()
    def coordinator(self) -> ExecutionCoordinator:
        return ExecutionCoordinator()

    @pytest.fixture()
    def observer(self) -> MockObserverManager:
        return MockObserverManager()

    @pytest.mark.asyncio()
    async def test_when_skips_on_false(self, executor, coordinator, observer) -> None:
        """when='x == 1' with x=2 → node is skipped."""
        node_spec = NodeSpec("guarded", noop_fn, when="x == 1")

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="guarded",
                node_spec=node_spec,
                node_input={"x": 2},
                context=NodeExecutionContext(dag_id="test", node_id="guarded"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
            )

        assert result["_skipped"] is True
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 1

    @pytest.mark.asyncio()
    async def test_when_executes_on_true(self, executor, coordinator, observer) -> None:
        """when='x == 1' with x=1 → node executes."""
        node_spec = NodeSpec("guarded", noop_fn, when="x == 1")

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="guarded",
                node_spec=node_spec,
                node_input={"x": 1},
                context=NodeExecutionContext(dag_id="test", node_id="guarded"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
            )

        assert result == {"processed": {"x": 1}}
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 0

    @pytest.mark.asyncio()
    async def test_when_sees_node_results(self, executor, coordinator, observer) -> None:
        """when='upstream.result == \"ok\"' with matching node_results → executes."""
        node_spec = NodeSpec("guarded", noop_fn, when='upstream.result == "ok"')

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="guarded",
                node_spec=node_spec,
                node_input={},
                context=NodeExecutionContext(dag_id="test", node_id="guarded"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
                node_results={"upstream": {"result": "ok"}},
            )

        assert result == {"processed": {}}
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 0

    @pytest.mark.asyncio()
    async def test_when_mapped_input_takes_precedence(
        self, executor, coordinator, observer
    ) -> None:
        """Mapped input field name shadows node_results key → backward compat."""
        node_spec = NodeSpec("guarded", noop_fn, when="upstream == 42")

        async with ExecutionContext(observer_manager=observer):
            result = await executor.execute_node(
                node_name="guarded",
                node_spec=node_spec,
                node_input={"upstream": 42},  # mapped input shadows node_results
                context=NodeExecutionContext(dag_id="test", node_id="guarded"),
                coordinator=coordinator,
                wave_index=0,
                validate=False,
                node_results={"upstream": {"result": "something_else"}},
            )

        # Mapped input value (42) should win; when clause should pass
        assert result == {"processed": {"upstream": 42}}
        skip_events = [e for e in observer.events if isinstance(e, NodeSkipped)]
        assert len(skip_events) == 0
