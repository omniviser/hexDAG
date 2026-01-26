"""Node executor for individual node execution.

This module provides the NodeExecutor class that handles executing individual
nodes with full lifecycle management including validation, timeout, and events.
"""

import asyncio
import contextvars
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort
    from hexdag.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexdag.core.context import get_observer_manager, get_policy_manager, set_current_node_name
from hexdag.core.domain.dag import NodeSpec, ValidationError
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.components.policy_coordinator import (
    OrchestratorError,
    PolicyCoordinator,
)
from hexdag.core.orchestration.events import NodeCancelled, NodeCompleted, NodeFailed, NodeStarted
from hexdag.core.orchestration.models import NodeExecutionContext
from hexdag.core.orchestration.policies.models import PolicySignal

logger = get_logger(__name__)


class NodeExecutionError(Exception):
    """Exception raised when a node fails to execute."""

    def __init__(self, node_name: str, original_error: Exception) -> None:
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")


class NodeTimeoutError(NodeExecutionError):
    """Exception raised when a node exceeds its timeout."""

    def __init__(self, node_name: str, timeout: float, original_error: TimeoutError) -> None:
        self.timeout = timeout
        super().__init__(node_name, original_error)


class NodeExecutor:
    """Handles individual node execution with validation, timeout, and retry logic.

    This component is responsible for executing a single node with full lifecycle
    management:

    - **Input validation**: Validates input data using node's input model
    - **Event emission**: Fires NodeStarted, NodeCompleted, NodeFailed events
    - **Policy evaluation**: Checks policies before and after execution
    - **Timeout handling**: Enforces per-node and global timeouts
    - **Output validation**: Validates output data using node's output model
    - **Error handling**: Converts exceptions to NodeExecutionError

    Single Responsibility: Execute a single node with all its lifecycle concerns.

    Examples
    --------
    Example usage::

        executor = NodeExecutor(strict_validation=True, default_node_timeout=30.0)

        result = await executor.execute_node(
            node_name="my_node",
            node_spec=NodeSpec("my_node", my_function),
            node_input={"data": "value"},
            ports={"llm": llm_adapter},
            context=execution_context,
            policy_coordinator=coordinator,
            observer_manager=observer,
            policy_manager=policy,
            wave_index=0
        )
    """

    def __init__(
        self,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
    ) -> None:
        """Initialize node executor.

        Parameters
        ----------
        strict_validation : bool, default=False
            If True, raise errors on validation failure.
            If False, log warnings and continue with unvalidated data.
        default_node_timeout : float | None, default=None
            Default timeout in seconds for each node.
            Can be overridden per-node via NodeSpec.timeout.
            None means no timeout.

        Examples
        --------
        Strict validation, 30 second default timeout::

            executor = NodeExecutor(strict_validation=True, default_node_timeout=30.0)

        Lenient validation, no timeout::

            executor = NodeExecutor(strict_validation=False)
        """
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

    async def execute_node(
        self,
        node_name: str,
        node_spec: NodeSpec,
        node_input: Any,
        context: NodeExecutionContext,
        policy_coordinator: PolicyCoordinator,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node with full lifecycle management.
        ...     wave_index=1
        ... )
        """
        node_start_time = time.time()

        observer_mgr = get_observer_manager()
        policy_mgr = get_policy_manager()

        try:
            # Input validation
            if validate:
                try:
                    validated_input = node_spec.validate_input(node_input)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    logger.debug(
                        "Input validation failed for node '{node}': {error}",
                        node=node_name,
                        error=e,
                    )
                    validated_input = node_input
            else:
                validated_input = node_input

            # Set current node name for port-level event attribution
            set_current_node_name(node_name)

            # Fire node started event and check policy
            start_event = NodeStarted(
                name=node_name,
                wave_index=wave_index,
                dependencies=tuple(node_spec.deps),
            )
            await policy_coordinator.notify_observer(observer_mgr, start_event)

            # Evaluate policy for node start
            start_response = await policy_coordinator.evaluate_policy(
                policy_mgr, start_event, context, node_id=node_name, wave_index=wave_index
            )

            if start_response.signal == PolicySignal.SKIP:
                return start_response.data
            if start_response.signal == PolicySignal.FAIL:
                raise OrchestratorError(f"Node '{node_name}' blocked: {start_response.data}")
            if start_response.signal != PolicySignal.PROCEED:
                raise OrchestratorError(
                    f"Node '{node_name}' blocked: {start_response.signal.value}"
                )

            # Determine timeout: node_spec.timeout > orchestrator default
            node_timeout = node_spec.timeout or self.default_node_timeout

            # Determine max retries: node_spec.max_retries or 1 (no retries)
            max_retries = node_spec.max_retries or 1
            last_error: Exception | None = None
            raw_output: Any = None  # Initialize to satisfy type checker

            for attempt in range(1, max_retries + 1):
                try:
                    if node_timeout:
                        async with asyncio.timeout(node_timeout):
                            raw_output = await self._execute_function(
                                node_spec, validated_input, kwargs
                            )
                    else:
                        raw_output = await self._execute_function(
                            node_spec, validated_input, kwargs
                        )
                    break  # Success - exit retry loop
                except TimeoutError as e:
                    # node_timeout is guaranteed to be set here because TimeoutError
                    # only occurs when timeout is set
                    timeout_value = node_timeout if node_timeout is not None else 0.0
                    last_error = NodeTimeoutError(node_name, timeout_value, e)
                    if attempt < max_retries:
                        logger.debug(
                            "Node '{node}' timeout ({attempt}/{max_retries}), retrying...",
                            node=node_name,
                            attempt=attempt,
                            max_retries=max_retries,
                        )
                        continue
                    raise last_error from e
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.debug(
                            "Node '{node}' error ({attempt}/{max_retries}): {error}, retrying...",
                            node=node_name,
                            attempt=attempt,
                            max_retries=max_retries,
                            error=e,
                        )
                        continue
                    raise
            else:
                # Loop completed without break - should not happen but handle it
                if last_error:
                    raise last_error

            # Output validation
            if validate:
                try:
                    validated_output = node_spec.validate_output(raw_output)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    logger.debug(
                        "Output validation failed for node '{node}': {error}",
                        node=node_name,
                        error=e,
                    )
                    validated_output = raw_output
            else:
                validated_output = raw_output

            # Fire node completed event
            complete_event = NodeCompleted(
                name=node_name,
                wave_index=wave_index,
                result=validated_output,
                duration_ms=(time.time() - node_start_time) * 1000,
            )
            await policy_coordinator.notify_observer(observer_mgr, complete_event)

            # Clear current node name
            set_current_node_name(None)

            return validated_output

        except NodeTimeoutError:
            # Node timed out - emit cancelled event and re-raise
            cancel_event = NodeCancelled(
                name=node_name,
                wave_index=wave_index,
                reason="timeout",
            )
            await policy_coordinator.notify_observer(observer_mgr, cancel_event)
            set_current_node_name(None)  # Clear on timeout
            raise  # Re-raise original timeout error

        except NodeExecutionError:
            # Already wrapped - just re-raise
            set_current_node_name(None)  # Clear on error
            raise

        except (ValidationError, ValueError, TypeError, KeyError, AttributeError) as validation_err:
            # Validation/type errors - emit failure event
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=validation_err,
            )
            await policy_coordinator.notify_observer(observer_mgr, fail_event)

            # Evaluate policy for validation failure
            fail_response = await policy_coordinator.evaluate_policy(
                policy_mgr, fail_event, context, node_id=node_name, wave_index=wave_index
            )

            if fail_response.signal == PolicySignal.RETRY:
                raise  # Let orchestrator retry

            # Wrap and propagate
            set_current_node_name(None)  # Clear on validation error
            raise NodeExecutionError(node_name, validation_err) from validation_err

        except RuntimeError as runtime_err:
            # Runtime execution errors
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=runtime_err,
            )
            await policy_coordinator.notify_observer(observer_mgr, fail_event)

            # Evaluate policy
            fail_response = await policy_coordinator.evaluate_policy(
                policy_mgr, fail_event, context, node_id=node_name, wave_index=wave_index
            )

            if fail_response.signal == PolicySignal.RETRY:
                raise

            set_current_node_name(None)  # Clear on runtime error
            raise NodeExecutionError(node_name, runtime_err) from runtime_err

    async def _execute_function(
        self,
        node_spec: NodeSpec,
        validated_input: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute node function. Ports accessed via ExecutionContext, not parameters."""
        if asyncio.iscoroutinefunction(node_spec.fn):
            return await node_spec.fn(validated_input, **kwargs)
        # Run sync functions in executor to avoid blocking event loop
        # IMPORTANT: Copy context so ContextVars propagate to thread pool
        ctx = contextvars.copy_context()

        def _run_sync() -> Any:
            return node_spec.fn(validated_input, **kwargs)

        return await asyncio.get_running_loop().run_in_executor(None, ctx.run, _run_sync)
