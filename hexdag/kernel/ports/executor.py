"""Port interface for DAG execution strategies.

This port enables pluggable execution backends for the orchestrator,
allowing the same DAG to run locally (in-process), with Celery
(distributed task queue), or Azure Functions (serverless).
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from hexdag.kernel.orchestration.hook_context import PipelineStatus


class ExecutionTask(BaseModel):
    """A serializable task definition for node execution.

    This model encapsulates all information needed to execute a node,
    allowing it to be passed across process boundaries for distributed execution.

    Attributes
    ----------
    node_name : str
        Unique identifier for the node within the DAG
    node_input : Any
        Input data for the node (None = executor should prepare from context)
    wave_index : int
        Wave number in the DAG execution sequence
    should_validate : bool
        Whether to perform input/output validation
    context_data : dict[str, Any]
        Execution context metadata (dag_id, run_id, attempt, etc.)
    params : dict[str, Any]
        Additional parameters passed to the node function
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    node_name: str
    node_input: Any | None = None
    wave_index: int = 0
    should_validate: bool = Field(default=True, validation_alias="validate")
    context_data: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of a node execution.

    Attributes
    ----------
    node_name : str
        Name of the executed node
    output : Any
        Result data from the node execution
    duration_ms : float
        Execution time in milliseconds
    status : PipelineStatus
        Execution status (SUCCESS, FAILED, CANCELLED)
    error : str | None
        Error message if execution failed, None otherwise
    error_type : str | None
        Type of error that occurred, if any
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_name: str
    output: Any = None
    duration_ms: float = 0.0
    status: PipelineStatus = PipelineStatus.SUCCESS
    error: str | None = None
    error_type: str | None = None


@runtime_checkable
class Executor(Protocol):
    """Port interface for DAG node execution strategies.

    This port abstracts the execution backend, allowing the orchestrator
    to delegate node execution to different implementations:

    - **LocalExecutor**: In-process async execution (default)
    - **CeleryExecutor**: Distributed execution via Celery task queue
    - **AzureFunctionsExecutor**: Serverless execution via Azure Functions

    The port provides a consistent interface regardless of where/how
    nodes are actually executed.

    Lifecycle
    ---------
    Executors may implement optional setup/cleanup methods:
    - asetup(): Initialize resources (connections, workers, etc.)
    - aclose(): Cleanup resources (called automatically by orchestrator)

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - asetup(): Initialize executor resources before first use
    - aclose(): Cleanup executor resources after execution completes
    """

    @abstractmethod
    async def aexecute_node(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a single node and return the result.

        The executor is responsible for:
        1. Resolving the node function (by name or from task data)
        2. Executing the function with the provided input
        3. Handling timeouts and errors
        4. Wrapping the result in ExecutionResult

        Parameters
        ----------
        task : ExecutionTask
            Task definition containing all execution parameters

        Returns
        -------
        ExecutionResult
            Execution result with output or error information

        Examples
        --------
        Local execution::

            task = ExecutionTask(
                node_name="fetch_data",
                node_input={"url": "https://api.example.com"},
                wave_index=0,
                validate=True
            )
            result = await executor.aexecute_node(task)
            if result.status == PipelineStatus.SUCCESS:
                print(result.output)

        Distributed execution (same interface)::

            # Task is serialized and sent to remote worker
            result = await celery_executor.aexecute_node(task)
        """
        ...

    @abstractmethod
    async def aexecute_wave(self, tasks: list[ExecutionTask]) -> dict[str, ExecutionResult]:
        """Execute multiple nodes concurrently within a wave.

        The executor should run all tasks in parallel (respecting concurrency
        limits) and return results as a dictionary.

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
        Exception
            If any task fails, the exception should be captured in
            ExecutionResult.error rather than raised directly

        Examples
        --------
        Execute wave of independent nodes::

            tasks = [
                ExecutionTask(node_name="fetch_a", ...),
                ExecutionTask(node_name="fetch_b", ...),
                ExecutionTask(node_name="fetch_c", ...),
            ]
            results = await executor.aexecute_wave(tasks)
            # All three nodes executed concurrently
            assert "fetch_a" in results
            assert "fetch_b" in results
            assert "fetch_c" in results
        """
        ...

    async def asetup(self) -> None:
        """Initialize executor resources before first use (optional).

        Implementations should use this to:
        - Establish connections to brokers/queues
        - Initialize worker pools
        - Verify remote endpoints are accessible
        - Load configuration

        This method is called automatically by the orchestrator during
        the managed_ports lifecycle.

        Examples
        --------
        Celery executor setup::

            async def asetup(self):
                self.celery_app = Celery('hexdag', broker=self.broker_url)
                await self._verify_broker_connection()

        Azure Functions executor setup::

            async def asetup(self):
                self.http_client = aiohttp.ClientSession()
                await self._verify_function_endpoint()
        """
        ...

    async def aclose(self) -> None:
        """Cleanup executor resources after execution completes (optional).

        Implementations should use this to:
        - Close connections to brokers/queues
        - Shutdown worker pools
        - Close HTTP clients
        - Flush any pending tasks

        This method is called automatically by the orchestrator during
        cleanup phase.

        Examples
        --------
        Celery executor cleanup::

            async def aclose(self):
                await self.celery_app.close()

        Azure Functions executor cleanup::

            async def aclose(self):
                await self.http_client.close()
        """
        ...


# Backward-compat alias (deprecated: use Executor)
ExecutorPort = Executor
