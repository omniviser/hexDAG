"""WaveExecutor - Handles parallel execution of nodes within waves.

Extracted from Orchestrator to provide a focused component for concurrent
wave-based execution with semaphore-based concurrency limiting.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.ports.observer_manager import ObserverManagerPort
    from hexai.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

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
    >>> executor = WaveExecutor(max_concurrent_nodes=10)
    >>> cancelled = await executor.execute_all_waves(
    ...     waves=waves,
    ...     node_executor_fn=orchestrator._execute_node,
    ...     ...
    ... )
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
        all_ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        policy_manager: PolicyManagerPort | None,
        policy_coordinator: Any,  # PolicyCoordinator
        timeout: float | None,
        validate: bool,
        **kwargs: Any,
    ) -> bool:
        """Execute all waves with optional timeout.

        Parameters
        ----------
        waves : list[list[str]]
            List of waves (each wave is a list of node names)
        node_executor_fn : Callable
            Function to execute a single node (typically orchestrator._execute_node)
        graph : DirectedGraph
            The DirectedGraph being executed
        node_results : dict[str, Any]
            Dictionary to accumulate node results (mutated in place)
        initial_input : Any
            Initial input data
        all_ports : dict[str, Any]
            All available ports
        context : NodeExecutionContext
            Execution context
        observer_manager : ObserverManagerPort | None
            Observer manager
        policy_manager : PolicyManagerPort | None
            Policy manager
        policy_coordinator : PolicyCoordinator
            Policy coordinator for evaluating policies
        timeout : float | None
            Optional timeout in seconds for all waves
        validate : bool
            Whether to validate nodes
        **kwargs : Any
            Additional arguments passed to node executor

        Returns
        -------
        bool
            True if execution was cancelled due to timeout, False otherwise
        """
        try:
            async with asyncio.timeout(timeout):
                for wave_idx, wave in enumerate(waves, 1):
                    wave_start_time = time.time()

                    # Fire wave started event and check control
                    wave_event = WaveStarted(
                        wave_index=wave_idx,
                        nodes=wave,
                    )
                    await policy_coordinator.notify_observer(observer_manager, wave_event)

                    # Evaluate policy for wave
                    wave_response = await policy_coordinator.evaluate_policy(
                        policy_manager, wave_event, context, wave_index=wave_idx
                    )
                    policy_coordinator.check_policy_signal(wave_response, f"Wave {wave_idx}")

                    # Execute the wave (node timeouts raise NodeTimeoutError, not TimeoutError)
                    wave_results = await self._execute_wave(
                        wave=wave,
                        node_executor_fn=node_executor_fn,
                        graph=graph,
                        node_results=node_results,
                        initial_input=initial_input,
                        ports=all_ports,
                        context=context,
                        observer_manager=observer_manager,
                        policy_manager=policy_manager,
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
                    await policy_coordinator.notify_observer(observer_manager, wave_completed)

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
        ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        policy_manager: PolicyManagerPort | None,
        wave_index: int,
        validate: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute all nodes in a wave with concurrency limiting.

        Parameters
        ----------
        wave : list[str]
            List of node names to execute in parallel
        node_executor_fn : Callable
            Function to execute a single node
        graph : DirectedGraph
            The DirectedGraph being executed
        node_results : dict[str, Any]
            Results from previous nodes
        initial_input : Any
            Initial input data
        ports : dict[str, Any]
            Available ports
        context : NodeExecutionContext
            Execution context
        observer_manager : ObserverManagerPort | None
            Observer manager
        policy_manager : PolicyManagerPort | None
            Policy manager
        wave_index : int
            Current wave index
        validate : bool
            Whether to validate nodes
        **kwargs : Any
            Additional arguments

        Returns
        -------
        dict[str, Any]
            Dictionary mapping node names to their execution results for this wave
        """

        async def execute_with_semaphore(node_name: str) -> tuple[str, Any]:
            """Execute a single node with semaphore-based concurrency control."""
            async with self._semaphore:
                result = await node_executor_fn(
                    node_name=node_name,
                    graph=graph,
                    node_results=node_results,
                    initial_input=initial_input,
                    ports=ports,
                    context=context,
                    observer_manager=observer_manager,
                    policy_manager=policy_manager,
                    wave_index=wave_index,
                    validate=validate,
                    **kwargs,
                )
                return node_name, result

        # Execute all nodes in the wave concurrently
        tasks = [execute_with_semaphore(node_name) for node_name in wave]
        wave_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        wave_results = {}
        for output in wave_outputs:
            if isinstance(output, BaseException):
                # Re-raise exceptions (they will be caught by the orchestrator)
                raise output

            node_name, result = output
            wave_results[node_name] = result

        return wave_results
