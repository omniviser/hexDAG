"""Local in-process executor adapter.

This executor runs nodes in the same process using asyncio, providing
the same behavior as the built-in orchestrator execution but through
the Executor interface.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from hexdag.kernel.context import get_port
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.components import (
    ExecutionCoordinator,
    NodeExecutionError,
    NodeExecutor,
)
from hexdag.kernel.orchestration.constants import (
    EXECUTOR_CONTEXT_GRAPH,
    EXECUTOR_CONTEXT_INITIAL_INPUT,
    EXECUTOR_CONTEXT_NODE_RESULTS,
)
from hexdag.kernel.orchestration.hook_context import PipelineStatus
from hexdag.kernel.orchestration.models import NodeExecutionContext
from hexdag.kernel.ports.executor import (
    ExecutionResult,
    ExecutionTask,
)

if TYPE_CHECKING:
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)


def _calculate_duration_ms(start_time: float) -> float:
    """Calculate duration in milliseconds from start time.

    Parameters
    ----------
    start_time : float
        Start time from time.time()

    Returns
    -------
    float
        Duration in milliseconds
    """
    return (time.time() - start_time) * 1000


class Local:
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


class LocalExecutor:
    """Local in-process executor using asyncio.

    This executor wraps the core NodeExecutor to provide
    the same in-process execution behavior but through the Executor
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

    def __init__(
        self, strict_validation: bool = True, default_node_timeout: float = 60.0, **kwargs: Any
    ) -> None:
        """Initialize local executor.

        Parameters
        ----------
        strict_validation : bool, default=True
            Enable strict validation
        default_node_timeout : float, default=60.0
            Default timeout for nodes in seconds
        """
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout
        self.max_concurrent_nodes = kwargs.get("max_concurrent_nodes", 10)

        # Core execution components
        self._node_executor = NodeExecutor(
            strict_validation=strict_validation,
            default_node_timeout=default_node_timeout,
        )
        self._execution_coordinator = ExecutionCoordinator()

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

        Raises
        ------
        RuntimeError
            If executor not initialized via asetup()

        Notes
        -----
        This method expects that the orchestrator has set up the execution
        context with the graph and ports. The task.context_data should contain
        the necessary execution context information.
        """
        if not self._initialized:
            raise RuntimeError("LocalExecutor not initialized - call asetup() first")

        start_time = time.time()

        try:
            # Note: Uses ContextVar pattern consistent with rest of codebase
            graph: DirectedGraph = get_port(EXECUTOR_CONTEXT_GRAPH)
            node_results: dict[str, Any] = get_port(EXECUTOR_CONTEXT_NODE_RESULTS)
            initial_input: Any = get_port(EXECUTOR_CONTEXT_INITIAL_INPUT)

            if graph is None:
                raise RuntimeError(
                    "LocalExecutor requires execution context with graph. "
                    "Ensure orchestrator has set up context properly."
                )

            if task.node_name not in graph.nodes:
                return ExecutionResult(
                    node_name=task.node_name,
                    status=PipelineStatus.FAILED,
                    error=f"Node '{task.node_name}' not found in graph",
                    error_type="KeyError",
                    duration_ms=_calculate_duration_ms(start_time),
                )

            node_spec: NodeSpec = graph.nodes[task.node_name]

            # Prepare input using ExecutionCoordinator
            node_input = self._execution_coordinator.prepare_node_input(
                node_spec, node_results, initial_input
            )

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
                coordinator=self._execution_coordinator,
                wave_index=task.wave_index,
                validate=task.should_validate,
                **task.params,
            )

            return ExecutionResult(
                node_name=task.node_name,
                output=output,
                status=PipelineStatus.SUCCESS,
                duration_ms=_calculate_duration_ms(start_time),
            )

        except NodeExecutionError:
            # NodeExecutionError should propagate directly (orchestrator will handle)
            raise
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            # Catch expected execution errors (validation, type issues, missing data)
            logger.error(f"LocalExecutor: Node '{task.node_name}' failed: {e}")
            return ExecutionResult(
                node_name=task.node_name,
                status=PipelineStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=_calculate_duration_ms(start_time),
            )
        except Exception as e:
            # Catch unexpected errors but log with higher severity
            logger.exception(f"LocalExecutor: Unexpected error in node '{task.node_name}': {e}")
            return ExecutionResult(
                node_name=task.node_name,
                status=PipelineStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=_calculate_duration_ms(start_time),
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

        Raises
        ------
        RuntimeError
            If executor not initialized via asetup()

        Notes
        -----
        This method uses asyncio.gather to execute all tasks concurrently,
        respecting the max_concurrent_nodes limit via semaphore.
        """
        if not self._initialized:
            raise RuntimeError("LocalExecutor not initialized - call asetup() first")

        semaphore = asyncio.Semaphore(self.max_concurrent_nodes)

        async def execute_with_limit(task: ExecutionTask) -> ExecutionResult:
            """Execute task with semaphore-based concurrency control."""
            async with semaphore:
                return await self.aexecute_node(task)

        # Execute all tasks concurrently with semaphore limit
        results_list = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks],
            return_exceptions=True,
        )

        results = {}
        for result in results_list:
            if isinstance(result, ExecutionResult):
                results[result.node_name] = result
            elif isinstance(result, Exception):
                # Can happen if task is cancelled during gather()
                logger.error(f"Exception during wave execution: {result}")
                raise result

        return results

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
