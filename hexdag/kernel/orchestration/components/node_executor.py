"""Node executor for individual node execution.

This module provides the NodeExecutor class that handles executing individual
nodes with full lifecycle management including validation, timeout, and events.
"""

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel.ports.observer_manager import ObserverManagerPort
else:
    ObserverManagerPort = Any

from hexdag.kernel.context import get_observer_manager, set_current_node_name
from hexdag.kernel.domain.dag import NodeSpec, NodeValidationError
from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.expression_parser import ExpressionError, compile_expression
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.components.execution_coordinator import ExecutionCoordinator
from hexdag.kernel.orchestration.events import (
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeStarted,
)
from hexdag.kernel.orchestration.models import NodeExecutionContext
from hexdag.kernel.utils.node_timer import Timer
from hexdag.kernel.validation.retry import RetryConfig, execute_with_retry

logger = get_logger(__name__)


class NodeExecutionError(HexDAGError):
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
    - **Timeout handling**: Enforces per-node and global timeouts
    - **Retry logic**: Exponential backoff retry on failure
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
            context=execution_context,
            coordinator=coordinator,
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
        coordinator: ExecutionCoordinator,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node with full lifecycle management.

        Parameters
        ----------
        node_name : str
            Name of the node being executed.
        node_spec : NodeSpec
            Node specification containing function and validation.
        node_input : Any
            Input data for the node.
        context : NodeExecutionContext
            Execution context with ports and configuration.
        coordinator : ExecutionCoordinator
            Coordinator for observer notifications.
        wave_index : int, default=0
            Index of the execution wave.
        validate : bool, default=True
            Whether to validate input/output.
        **kwargs : Any
            Additional keyword arguments passed to the node function.

        Returns
        -------
        Any
            The validated output from the node execution.
        """
        node_timer = Timer()

        observer_mgr = get_observer_manager()

        try:
            # Input validation
            if validate:
                try:
                    validated_input = node_spec.validate_input(node_input)
                except NodeValidationError as e:
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

            # Auto-skip if all upstream dependencies were skipped
            if isinstance(validated_input, dict) and validated_input.get("_upstream_skipped"):
                logger.info(
                    "Node '{node}' skipped: all upstream dependencies were skipped",
                    node=node_name,
                )
                skip_event = NodeSkipped(
                    name=node_name,
                    wave_index=wave_index,
                    reason="all upstream dependencies were skipped",
                )
                await coordinator.notify_observer(observer_mgr, skip_event)
                return {
                    "_skipped": True,
                    "_upstream_skipped": True,
                    "reason": "all upstream dependencies were skipped",
                }

            # Evaluate when clause - skip node if condition evaluates to False
            if node_spec.when:
                try:
                    predicate = compile_expression(node_spec.when)
                    # Build data context from validated input
                    data_context = validated_input if isinstance(validated_input, dict) else {}
                    condition_result = predicate(data_context, {})

                    if not condition_result:
                        logger.info(
                            "Node '{node}' skipped: when clause '{when}' evaluated to False",
                            node=node_name,
                            when=node_spec.when,
                        )
                        # Emit NodeSkipped event
                        skip_event = NodeSkipped(
                            name=node_name,
                            wave_index=wave_index,
                            reason=f"when clause '{node_spec.when}' evaluated to False",
                        )
                        await coordinator.notify_observer(observer_mgr, skip_event)

                        return {
                            "_skipped": True,
                            "reason": f"when clause '{node_spec.when}' evaluated to False",
                        }
                except ExpressionError as e:
                    logger.error(
                        "Invalid when clause expression for node '{node}': {error}",
                        node=node_name,
                        error=e,
                    )
                    raise NodeExecutionError(
                        node_name, ValueError(f"Invalid when clause: {e}")
                    ) from e

            # Set current node name for port-level event attribution
            set_current_node_name(node_name)

            # Fire node started event
            start_event = NodeStarted(
                name=node_name,
                wave_index=wave_index,
                dependencies=tuple(node_spec.deps),
            )
            await coordinator.notify_observer(observer_mgr, start_event)

            # Determine timeout: node_spec.timeout > orchestrator default
            node_timeout = node_spec.timeout or self.default_node_timeout

            # Build retry config from node spec fields
            retry_config = RetryConfig.from_node_spec_fields(
                max_retries=node_spec.max_retries,
                retry_delay=node_spec.retry_delay,
                retry_backoff=node_spec.retry_backoff,
                retry_max_delay=node_spec.retry_max_delay,
            )

            async def _run_with_timeout() -> Any:
                if node_timeout:
                    async with asyncio.timeout(node_timeout):
                        return await self._execute_function(node_spec, validated_input, kwargs)
                return await self._execute_function(node_spec, validated_input, kwargs)

            def _on_retry(attempt: int, max_retries: int, error: Exception, delay: float) -> None:
                logger.debug(
                    "Node '{node}' error ({attempt}/{max_retries}): {error}, "
                    "retrying in {delay:.2f}s...",
                    node=node_name,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=error,
                    delay=delay,
                )

            try:
                raw_output = await execute_with_retry(
                    _run_with_timeout, retry_config, on_retry=_on_retry
                )
            except TimeoutError as e:
                timeout_value = node_timeout if node_timeout is not None else 0.0
                raise NodeTimeoutError(node_name, timeout_value, e) from e

            # Output validation
            if validate:
                try:
                    validated_output = node_spec.validate_output(raw_output)
                except NodeValidationError as e:
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
                duration_ms=node_timer.duration_ms,
            )
            await coordinator.notify_observer(observer_mgr, complete_event)

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
            await coordinator.notify_observer(observer_mgr, cancel_event)
            set_current_node_name(None)  # Clear on timeout
            raise  # Re-raise original timeout error

        except NodeExecutionError:
            # Already wrapped - just re-raise
            set_current_node_name(None)  # Clear on error
            raise

        except (
            NodeValidationError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as validation_err:
            # Validation/type errors - emit failure event
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=validation_err,
            )
            await coordinator.notify_observer(observer_mgr, fail_event)

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
            await coordinator.notify_observer(observer_mgr, fail_event)

            set_current_node_name(None)  # Clear on runtime error
            raise NodeExecutionError(node_name, runtime_err) from runtime_err

        except Exception as err:
            # Catch-all for any remaining errors (e.g. ParseError, HexDAGError
            # subclasses) so every node error carries node name context.
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=err,
            )
            await coordinator.notify_observer(observer_mgr, fail_event)

            set_current_node_name(None)
            raise NodeExecutionError(node_name, err) from err

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
