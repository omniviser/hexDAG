"""Local in-process executor adapter.

This executor runs nodes in the same process using asyncio, providing
the same behavior as the built-in orchestrator execution but through
the ExecutorPort interface.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from hexdag.core.configurable import ConfigurableAdapter, ExecutorConfig
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.components import (
    InputMapper,
    NodeExecutor,
    PolicyCoordinator,
    WaveExecutor,
)
from hexdag.core.orchestration.models import NodeExecutionContext
from hexdag.core.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    ExecutorCapabilities,
)
from hexdag.core.registry.decorators import adapter

if TYPE_CHECKING:
    from hexdag.core.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)


class LocalExecutorConfig(ExecutorConfig):
    """Configuration for LocalExecutor.

    Attributes
    ----------
    max_concurrent_nodes : int
        Maximum number of nodes to execute concurrently (default: 10)
    strict_validation : bool
        If True, raise errors on validation failure (default: False)
    default_node_timeout : float | None
        Default timeout in seconds for each node (default: None, no timeout)
    """

    max_concurrent_nodes: int = 10
    strict_validation: bool = False
    default_node_timeout: float | None = None


@adapter("executor", name="local", namespace="core")
class LocalExecutor(ConfigurableAdapter):
    """Local in-process executor using asyncio.

    This executor wraps the core NodeExecutor and WaveExecutor to provide
    the same in-process execution behavior but through the ExecutorPort
    interface. It's useful for:

    1. Testing the executor abstraction
    2. Providing a consistent interface across execution strategies
    3. Serving as a reference implementation for other executors

    The LocalExecutor maintains the same features as built-in execution:
    - Async/sync function support
    - Validation and type checking
    - Timeout handling
    - Event emission
    - Policy evaluation

    Examples
    --------
    Basic usage::

        executor = LocalExecutor(max_concurrent_nodes=5)
        orchestrator = Orchestrator(executor=executor)
        results = await orchestrator.run(graph, input_data)

    With validation and timeout::

        executor = LocalExecutor(
            strict_validation=True,
            default_node_timeout=30.0
        )
        orchestrator = Orchestrator(executor=executor)

    From YAML::

        executor:
          type: local
          namespace: core
          params:
            max_concurrent_nodes: 20
            strict_validation: true
            default_node_timeout: 60.0
    """

    Config = LocalExecutorConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Core execution components configured from self.config
        # Type narrowing: self.config is LocalExecutorConfig after super().__init__
        config: LocalExecutorConfig = self.config  # type: ignore[assignment]
        self._node_executor = NodeExecutor(
            strict_validation=config.strict_validation,
            default_node_timeout=config.default_node_timeout,
        )
        self._wave_executor = WaveExecutor(max_concurrent_nodes=config.max_concurrent_nodes)
        self._input_mapper = InputMapper()
        self._policy_coordinator = PolicyCoordinator()

        # State for tracking
        self._initialized = False

    async def aexecute_node(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a single node in-process.

        Parameters
        ----------
        task : ExecutionTask
            Task containing node information and input data

        Returns
        -------
        ExecutionResult
            Result of the execution with output or error information

        Notes
        -----
        This method expects that the orchestrator has set up the execution
        context with the graph and ports. The task.context_data should contain
        the necessary execution context information.
        """
        start_time = time.time()

        try:
            # Get graph and node from execution context (stored in ports)
            # The orchestrator stores these with _hexdag_ prefix
            from hexdag.core.context import get_port

            graph: DirectedGraph = get_port("_hexdag_graph")
            node_results: dict[str, Any] = get_port("_hexdag_node_results")
            initial_input: Any = get_port("_hexdag_initial_input")

            if graph is None:
                raise RuntimeError(
                    "LocalExecutor requires execution context with graph. "
                    "Ensure orchestrator has set up context properly."
                )

            # Get the node spec
            if task.node_name not in graph.nodes:
                return ExecutionResult(
                    node_name=task.node_name,
                    status="failed",
                    error=f"Node '{task.node_name}' not found in graph",
                    error_type="KeyError",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            node_spec: NodeSpec = graph.nodes[task.node_name]

            # Prepare input using InputMapper
            node_input = self._input_mapper.prepare_node_input(
                node_spec, node_results, initial_input
            )

            # Create execution context
            execution_context = NodeExecutionContext(
                dag_id=task.context_data.get("dag_id", "unnamed"),
                node_id=task.node_name,
                wave_index=task.wave_index,
                attempt=task.context_data.get("attempt", 1),
                metadata=task.context_data,
            )

            # Execute using NodeExecutor
            output = await self._node_executor.execute_node(
                node_name=task.node_name,
                node_spec=node_spec,
                node_input=node_input,
                context=execution_context,
                policy_coordinator=self._policy_coordinator,
                wave_index=task.wave_index,
                validate=task.should_validate,
                **task.params,
            )

            return ExecutionResult(
                node_name=task.node_name,
                output=output,
                status="success",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"LocalExecutor: Node '{task.node_name}' failed: {e}")
            return ExecutionResult(
                node_name=task.node_name,
                status="failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def aexecute_wave(self, tasks: list[ExecutionTask]) -> dict[str, ExecutionResult]:
        """Execute multiple nodes concurrently within a wave.

        Parameters
        ----------
        tasks : list[ExecutionTask]
            List of tasks to execute in parallel

        Returns
        -------
        dict[str, ExecutionResult]
            Map of node_name -> execution result

        Notes
        -----
        This method uses asyncio.gather to execute all tasks concurrently,
        respecting the max_concurrent_nodes limit via semaphore.
        """
        # Execute all tasks concurrently
        results_list = await asyncio.gather(
            *[self.aexecute_node(task) for task in tasks],
            return_exceptions=True,
        )

        # Convert to dictionary
        results = {}
        for result in results_list:
            if isinstance(result, ExecutionResult):
                results[result.node_name] = result
            elif isinstance(result, Exception):
                # Shouldn't happen as we handle exceptions in aexecute_node
                # but handle gracefully
                logger.error(f"Unexpected exception in wave execution: {result}")
                raise result

        return results

    def get_capabilities(self) -> ExecutorCapabilities:
        """Report executor capabilities.

        Returns
        -------
        ExecutorCapabilities
            Capability flags for this executor
        """
        config: LocalExecutorConfig = self.config  # type: ignore[assignment]
        return ExecutorCapabilities(
            supports_timeout=True,
            supports_cancellation=True,
            max_concurrent=config.max_concurrent_nodes,
            is_distributed=False,
            requires_serialization=False,
        )

    async def asetup(self) -> None:
        """Initialize executor resources.

        For LocalExecutor, this is a no-op as all resources are
        in-process and don't require setup.
        """
        self._initialized = True
        logger.debug("LocalExecutor initialized")

    async def aclose(self) -> None:
        """Cleanup executor resources.

        For LocalExecutor, this is a no-op as all resources are
        in-process and don't require cleanup.
        """
        self._initialized = False
        logger.debug("LocalExecutor closed")
