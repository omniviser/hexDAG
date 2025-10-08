"""WaveExecutor - Handles parallel execution of nodes within waves.

Extracted from Orchestrator to provide a focused component for concurrent
wave-based execution with semaphore-based concurrency limiting.
"""

import asyncio
import time
from typing import Any

from hexai.core.application.events import WaveCompleted, WaveStarted
from hexai.core.domain.dag import DirectedGraph
from hexai.core.orchestration.models import NodeExecutionContext


class WaveExecutor:
    """Executes waves of nodes concurrently with resource management.

    Responsibilities:
    - Execute all waves in topological order
    - Manage concurrency limits via semaphore
    - Handle timeout for overall wave execution
    - Emit wave-level events

    Parameters
    ----------
    max_concurrent_nodes : int
        Maximum number of nodes that can execute concurrently
    semaphore : asyncio.Semaphore | None
        Optional semaphore for concurrency control. If None, creates one.

    Examples
    --------
    Example usage::

        executor = WaveExecutor(max_concurrent_nodes=10)
        cancelled = await executor.execute_all_waves(
            waves=waves,
            node_executor_fn=orchestrator._execute_node,
            ...
        )
    """

    def __init__(
        self,
        max_concurrent_nodes: int = 10,
        semaphore: asyncio.Semaphore | None = None,
    ):
        self.max_concurrent_nodes = max_concurrent_nodes
        self._semaphore = semaphore or asyncio.Semaphore(max_concurrent_nodes)

    async def execute_all_waves(
        self,
        waves: list[list[str]],
        node_executor_fn: Any,  # Callable for executing a single node
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        context: NodeExecutionContext,
        policy_coordinator: Any,  # PolicyCoordinator
        timeout: float | None,
        validate: bool,
        **kwargs: Any,
    ) -> bool:
        """Execute all waves with optional timeout."""
        from hexai.core.context import (
            get_observer_manager,
            get_policy_manager,
        )

        try:
            async with asyncio.timeout(timeout):
                for wave_idx, wave in enumerate(waves, 1):
                    wave_start_time = time.time()

                    # Fire wave started event and check control
                    wave_event = WaveStarted(
                        wave_index=wave_idx,
                        nodes=wave,
                    )
                    await policy_coordinator.notify_observer(get_observer_manager(), wave_event)

                    # Evaluate policy for wave
                    wave_response = await policy_coordinator.evaluate_policy(
                        get_policy_manager(), wave_event, context, wave_index=wave_idx
                    )
                    policy_coordinator.check_policy_signal(wave_response, f"Wave {wave_idx}")

                    # Execute the wave (node timeouts raise NodeTimeoutError, not TimeoutError)
                    wave_results = await self._execute_wave(
                        wave=wave,
                        node_executor_fn=node_executor_fn,
                        graph=graph,
                        node_results=node_results,
                        initial_input=initial_input,
                        context=context,
                        wave_index=wave_idx,
                        validate=validate,
                        **kwargs,
                    )
                    node_results.update(wave_results)

                    # Fire wave completed event
                    wave_completed = WaveCompleted(
                        wave_index=wave_idx,
                        duration_ms=(time.time() - wave_start_time) * 1000,
                    )
                    await policy_coordinator.notify_observer(get_observer_manager(), wave_completed)

            return False  # Completed successfully

        except TimeoutError:
            # This is a pipeline-level timeout (from asyncio.timeout(timeout))
            # Node-level timeouts raise NodeTimeoutError and propagate up
            return True  # Cancelled due to timeout

    async def _execute_wave(
        self,
        wave: list[str],
        node_executor_fn: Any,
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        context: NodeExecutionContext,
        wave_index: int,
        validate: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute all nodes in a wave with concurrency limiting."""

        async def execute_with_semaphore(node_name: str) -> tuple[str, Any]:
            """Execute a single node with semaphore-based concurrency control."""
            async with self._semaphore:
                result = await node_executor_fn(
                    node_name=node_name,
                    graph=graph,
                    node_results=node_results,
                    initial_input=initial_input,
                    context=context,
                    wave_index=wave_index,
                    validate=validate,
                    **kwargs,
                )
                return node_name, result

        # Execute all nodes in the wave concurrently
        # Create coroutines
        coros = [execute_with_semaphore(node_name) for node_name in wave]

        # Note: asyncio.gather with return_exceptions=True handles BaseException propagation
        # internally and will cancel remaining tasks. We rely on the semaphore's async context
        # manager (__aexit__) to properly release resources even on cancellation.
        wave_outputs = await asyncio.gather(*coros, return_exceptions=True)

        # Process results and handle exceptions (lazy list creation)
        wave_results = {}
        exceptions: list[tuple[str | None, Exception]] | None = None

        for output in wave_outputs:
            if isinstance(output, Exception):
                # Lazy create exception list only when first exception found
                if exceptions is None:
                    exceptions = []
                exceptions.append((None, output))
            elif isinstance(output, BaseException):
                # For non-Exception BaseExceptions (KeyboardInterrupt, SystemExit),
                # these should have been caught above, but handle defensively
                raise output
            else:
                node_name, result = output
                wave_results[node_name] = result

        # Raise all collected exceptions
        if exceptions:
            if len(exceptions) == 1:
                # Single exception - raise directly
                raise exceptions[0][1]
            # Multiple exceptions - raise as ExceptionGroup
            exception_list = [exc for _, exc in exceptions]
            raise ExceptionGroup(
                f"Multiple node failures in wave {wave_index} ({len(exceptions)} nodes failed)",
                exception_list,
            )

        return wave_results
