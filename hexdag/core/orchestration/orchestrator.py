"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort
    from hexdag.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexdag.core.context import (
    ExecutionContext,
    get_observer_manager,
    get_ports,
    set_current_graph,
    set_node_results,
    set_ports,
)
from hexdag.core.domain.dag import DirectedGraph, DirectedGraphError
from hexdag.core.exceptions import OrchestratorError
from hexdag.core.logging import get_logger
from hexdag.core.orchestration import NodeExecutionContext
from hexdag.core.orchestration.components import (
    InputMapper,
    PolicyCoordinator,
)
from hexdag.core.orchestration.constants import (
    EXECUTOR_CONTEXT_GRAPH,
    EXECUTOR_CONTEXT_INITIAL_INPUT,
    EXECUTOR_CONTEXT_NODE_RESULTS,
)
from hexdag.core.orchestration.events import WaveCompleted, WaveStarted
from hexdag.core.orchestration.hook_context import PipelineStatus
from hexdag.core.orchestration.hooks import (
    HookConfig,
    PostDagHookConfig,
    PostDagHookManager,
    PreDagHookManager,
)
from hexdag.core.orchestration.models import PortsConfiguration
from hexdag.core.ports.executor import ExecutionTask, ExecutorPort
from hexdag.core.ports_builder import PortsBuilder

from .events import (
    PipelineCancelled,
    PipelineCompleted,
    PipelineStarted,
)
from .policies.models import PolicyResponse

logger = get_logger(__name__)


def _has_async_lifecycle(obj: Any, method_name: str) -> bool:
    """Check if object has an async lifecycle method (asetup/aclose).

    Parameters
    ----------
    obj : Any
        Object to check
    method_name : str
        Method name to check (e.g., "asetup", "aclose")

    Returns
    -------
    bool
        True if object has the method and it's a coroutine function
    """
    return hasattr(obj, method_name) and asyncio.iscoroutinefunction(
        getattr(obj, method_name, None)
    )


# Default configuration constants
DEFAULT_MAX_CONCURRENT_NODES = 10


@asynccontextmanager
async def _managed_ports(
    base_ports: dict[str, Any],
    additional_ports: dict[str, Any] | None = None,
    executor: ExecutorPort | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Manage port and executor lifecycle with automatic cleanup.

    Ports and executors that implement asetup()/aclose() methods will be
    automatically initialized and cleaned up.

    Parameters
    ----------
    base_ports : dict[str, Any]
        Base ports to manage
    additional_ports : dict[str, Any] | None
        Additional ports to merge with base ports
    executor : ExecutorPort | None
        Optional executor to manage lifecycle for
    """
    all_ports = {**base_ports}
    if additional_ports:
        all_ports.update(additional_ports)

    # Setup executor first (if provided)
    executor_initialized = False
    if executor is not None and _has_async_lifecycle(executor, "asetup"):
        try:
            await executor.asetup()
            executor_initialized = True
        except Exception as e:
            logger.error(f"Executor setup failed: {e}")
            raise

    # Setup ports
    initialized: list[str] = []
    for name, port in all_ports.items():
        if _has_async_lifecycle(port, "asetup"):
            try:
                await port.asetup()
                initialized.append(name)
            except Exception as e:
                logger.error(f"Port setup failed: {name}: {e}")
                # Cleanup initialized ports
                for cleanup_name in initialized:
                    cleanup_port = all_ports[cleanup_name]
                    if _has_async_lifecycle(cleanup_port, "aclose"):
                        with suppress(Exception):
                            await cleanup_port.aclose()
                # Cleanup executor if initialized
                if (
                    executor_initialized
                    and executor is not None
                    and _has_async_lifecycle(executor, "aclose")
                ):
                    with suppress(Exception):
                        await executor.aclose()
                raise

    try:
        yield all_ports
    finally:
        # Cleanup ports
        for name in initialized:
            port = all_ports[name]
            if _has_async_lifecycle(port, "aclose"):
                try:
                    await port.aclose()
                except Exception as e:
                    logger.warning(f"Port cleanup failed: {name}: {e}")

        # Cleanup executor last
        if (
            executor_initialized
            and executor is not None
            and _has_async_lifecycle(executor, "aclose")
        ):
            try:
                await executor.aclose()
            except Exception as e:
                logger.warning(f"Executor cleanup failed: {e}")


class Orchestrator:
    """Orchestrates DAG execution with concurrent processing and resource management.

    The orchestrator executes DirectedGraphs by:

    1. Computing execution waves via topological sorting
    2. Running each wave's nodes concurrently with configurable limits
    3. Passing outputs between nodes
    4. Tracking execution with events
    """

    def __init__(
        self,
        max_concurrent_nodes: int = DEFAULT_MAX_CONCURRENT_NODES,
        ports: Union[dict[str, Any], "PortsConfiguration", None] = None,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        pre_hook_config: HookConfig | None = None,
        post_hook_config: PostDagHookConfig | None = None,
        executor: ExecutorPort | None = None,
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Shared ports/dependencies for all pipeline executions.
                Can be either a flat dict (backward compatible) or a PortsConfiguration
                for advanced type-specific and node-level port customization.
            strict_validation: If True, raise errors on validation failure
            default_node_timeout: Default timeout in seconds for each node (None = no timeout)
            pre_hook_config: Configuration for pre-DAG hooks (health checks, secrets, etc.)
            post_hook_config: Configuration for post-DAG hooks (cleanup, checkpoints, etc.)
            executor: Optional executor port for pluggable execution strategies.
                If None (default), creates LocalExecutor with the provided configuration.
                Set to a custom ExecutorPort implementation (e.g., CeleryExecutor,
                AzureFunctionsExecutor) for distributed or serverless execution.

        Notes
        -----
        ARCHITECTURAL EXCEPTION: This is the ONLY place in hexdag/core that imports
        from hexdag/adapters (lazy import of LocalExecutor). This exception is
        enforced by pre-commit hooks to ensure it remains isolated.
        """

        self.max_concurrent_nodes = max_concurrent_nodes
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

        # Default to LocalExecutor if no executor provided
        # ARCHITECTURAL EXCEPTION: Lazy import to avoid core -> adapters dependency at module level
        if executor is None:
            from hexdag.adapters.executors import LocalExecutor  # noqa: PLC0415

            if default_node_timeout is not None:
                executor = LocalExecutor(
                    max_concurrent_nodes=max_concurrent_nodes,
                    strict_validation=strict_validation,
                    default_node_timeout=default_node_timeout,
                )
            else:
                executor = LocalExecutor(
                    max_concurrent_nodes=max_concurrent_nodes,
                    strict_validation=strict_validation,
                )

        self.executor = executor

        # Handle both dict and PortsConfiguration
        self.ports_config: PortsConfiguration | None
        if isinstance(ports, PortsConfiguration):
            self.ports_config = ports
            # Extract global ports for backward compatibility
            self.ports = (
                {k: v.port for k, v in ports.global_ports.items()} if ports.global_ports else {}
            )
        else:
            self.ports_config = None
            self.ports = ports or {}

        # Validate known port types
        self._validate_port_types(self.ports)

        # Initialize collaborators (dependency injection)
        self._policy_coordinator = PolicyCoordinator()
        self._input_mapper = InputMapper()

        # Initialize hook managers
        self._pre_hook_manager = PreDagHookManager(pre_hook_config)
        self._post_hook_manager = PostDagHookManager(post_hook_config, self._pre_hook_manager)

    async def _notify_observer(
        self, observer_manager: ObserverManagerPort | None, event: Any
    ) -> None:
        """Notify observer if it exists (delegates to PolicyCoordinator)."""
        await self._policy_coordinator.notify_observer(observer_manager, event)

    async def _evaluate_policy(
        self,
        policy_manager: PolicyManagerPort | None,
        event: Any,
        context: NodeExecutionContext,
        node_id: str | None = None,
        wave_index: int | None = None,
        attempt: int = 1,
    ) -> PolicyResponse:
        """Evaluate policy and create context (delegates to PolicyCoordinator)."""
        return await self._policy_coordinator.evaluate_policy(
            policy_manager, event, context, node_id, wave_index, attempt
        )

    def _check_policy_signal(self, response: PolicyResponse, context: str) -> None:
        """Check policy signal and raise error if not PROCEED (delegates to PolicyCoordinator)."""
        self._policy_coordinator.check_policy_signal(response, context)

    def _validate_port_types(self, ports: dict[str, Any]) -> None:
        """Validate that orchestrator ports match expected types.

        Args
        ----
            ports: Dictionary of ports to validate

        Notes
        -----
        This provides helpful warnings if ports don't match expected protocols.
        Currently checks observer_manager and policy_manager.
        """
        # Check observer_manager if provided
        if "observer_manager" in ports:
            obs = ports["observer_manager"]
            if not hasattr(obs, "notify"):
                logger.warning(
                    f"Port 'observer_manager' doesn't have 'notify' method. "
                    f"Expected ObserverManagerPort, got {type(obs).__name__}"
                )

        # Check policy_manager if provided
        if "policy_manager" in ports:
            policy = ports["policy_manager"]
            if not hasattr(policy, "evaluate"):
                logger.warning(
                    f"Port 'policy_manager' doesn't have 'evaluate' method. "
                    f"Expected PolicyManagerPort, got {type(policy).__name__}"
                )

    def _validate_required_ports(
        self, graph: DirectedGraph, available_ports: dict[str, Any]
    ) -> None:
        """Validate that all required ports for nodes in the DAG are available.

        Args
        ----
            graph: The DirectedGraph to validate
            available_ports: Dictionary of available ports

        Raises
        ------
            OrchestratorError: If required ports are missing
        """
        missing_ports: dict[str, list[str]] = {}

        for node_name, node_spec in graph.items():  # Using .items() instead of .nodes.items()
            # Check if node function has required_ports metadata
            fn = node_spec.fn
            required_ports: list[str] = []

            # Try to get required_ports from the function/method
            if hasattr(fn, "_hexdag_required_ports"):
                required_ports = getattr(fn, "_hexdag_required_ports", [])
            # Note: type ignore needed for pyright - hasattr doesn't narrow union types for it
            elif hasattr(fn, "__self__") and hasattr(fn.__self__, "__class__"):  # type: ignore[unused-ignore]
                # It's a bound method - check the class
                node_class = fn.__self__.__class__  # type: ignore[unused-ignore]
                required_ports = getattr(node_class, "_hexdag_required_ports", [])

            # Check each required port
            for port_name in required_ports:
                if port_name not in available_ports:
                    if node_name not in missing_ports:
                        missing_ports[node_name] = []
                    missing_ports[node_name].append(port_name)

        # Raise error if any ports are missing
        if missing_ports:
            error_msg = "Missing required ports:\n"
            for node_name, ports in missing_ports.items():
                error_msg += f"  Node '{node_name}': {', '.join(ports)}\n"
            error_msg += f"\nAvailable ports: {', '.join(available_ports.keys())}"
            raise OrchestratorError(error_msg)

    @classmethod
    def from_builder(
        cls,
        builder: PortsBuilder,
        max_concurrent_nodes: int = DEFAULT_MAX_CONCURRENT_NODES,
        strict_validation: bool = False,
        executor: ExecutorPort | None = None,
    ) -> "Orchestrator":
        """Create an Orchestrator using a PortsBuilder.

        This provides a more intuitive way to configure the orchestrator
        with type-safe port configuration.

        Args
        ----
            builder: Configured PortsBuilder instance
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            strict_validation: If True, raise errors on validation failure
            executor: Optional executor port for pluggable execution strategies

        Returns
        -------
            Orchestrator
                New orchestrator instance with configured ports

        Example
        -------
            ```python
            orchestrator = Orchestrator.from_builder(
                PortsBuilder()
                .with_llm(OpenAIAdapter())
                .with_database(PostgresAdapter())
                .with_observer_manager(LocalObserverManager())
                .with_policy_manager(LocalPolicyManager())
            )
            ```
        """
        return cls(
            max_concurrent_nodes=max_concurrent_nodes,
            ports=builder.build(),
            strict_validation=strict_validation,
            executor=executor,
        )

    async def run(
        self,
        graph: DirectedGraph,
        initial_input: Any,
        additional_ports: dict[str, Any] | PortsBuilder | None = None,
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a DAG with concurrent processing and resource limits.

        Supports both traditional dictionary-based ports and the new PortsBuilder
        for additional_ports parameter. When using PortsBuilder, it will be
        automatically converted to a dictionary before merging with base ports.

        Timeout behavior is controlled by PolicyManager. Policies can provide:
        - `timeout` in response.data dict for pipeline-level timeout
        - Checked at PipelineStarted event evaluation

        Args
        ----
            graph: The DirectedGraph to execute
            initial_input: Initial input data for the graph
            additional_ports: Either a dictionary of ports or a PortsBuilder instance
            validate: Whether to validate the graph before execution
            **kwargs: Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Dictionary mapping node names to their execution results

        Examples
        --------
        Using dictionary for additional ports (traditional approach):

        >>> results = await orchestrator.run(  # doctest: +SKIP
        ...     graph,
        ...     input_data,
        ...     additional_ports={"llm": MockLLM()}
        ... )

        Using PortsBuilder for additional ports (new approach):

        >>> results = await orchestrator.run(  # doctest: +SKIP
        ...     graph,
        ...     input_data,
        ...     additional_ports=PortsBuilder().with_llm(MockLLM())
        ... )
        """
        # Prepare additional ports (convert PortsBuilder if needed)
        additional_ports_dict: dict[str, Any] | None = None
        if additional_ports:
            if isinstance(additional_ports, PortsBuilder):
                additional_ports_dict = additional_ports.build()
            else:
                additional_ports_dict = additional_ports

        # Use managed_ports context manager for automatic lifecycle management
        async with _managed_ports(self.ports, additional_ports_dict, self.executor) as all_ports:
            return await self._execute_with_ports(
                graph, initial_input, all_ports, validate, **kwargs
            )

    async def _execute_with_ports(
        self,
        graph: DirectedGraph,
        initial_input: Any,
        all_ports: dict[str, Any],
        validate: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute DAG with managed ports (internal method).

        This method is separated to work with the managed_ports context manager.
        """
        if validate:
            # Validate DAG structure - catch specific DAG errors
            try:
                # By default, skip type checking for backward compatibility
                # Enable via graph.validate(check_type_compatibility=True)
                graph.validate(check_type_compatibility=False)
            except DirectedGraphError as e:
                # DAG-specific errors (cycles, missing nodes, etc.)
                raise OrchestratorError(f"Invalid DAG: {e}") from e
            except (ValueError, TypeError, KeyError) as e:
                # Other validation errors
                raise OrchestratorError(f"Invalid DAG: {e}") from e

        # Validate required ports for all nodes
        self._validate_required_ports(graph, all_ports)

        node_results: dict[str, Any] = {}
        waves = graph.waves()
        pipeline_start_time = time.time()

        # Get observer manager and control manager from ports
        # Get policy and observer managers from ports - expecting port interfaces
        observer_manager: ObserverManagerPort | None = all_ports.get("observer_manager")
        policy_manager: PolicyManagerPort | None = all_ports.get("policy_manager")

        # Create execution context for this DAG run
        pipeline_name = getattr(graph, "name", "unnamed")
        context = NodeExecutionContext(dag_id=pipeline_name)
        run_id = str(uuid.uuid4())

        # Set up execution context for all components
        async with ExecutionContext(
            observer_manager=observer_manager,
            policy_manager=policy_manager,
            run_id=run_id,
            ports=all_ports,
        ):
            # Set graph and node_results in context for dynamic expansion
            set_current_graph(graph)
            set_node_results(node_results)

            # PRE-DAG HOOKS: Execute before pipeline starts
            pre_hook_results = await self._pre_hook_manager.execute_hooks(
                context=context,
                pipeline_name=pipeline_name,
            )
            # Store hook results in context for nodes to access
            context.metadata["pre_dag_hooks"] = pre_hook_results

            # Fire pipeline started event and check control
            event = PipelineStarted(
                name=pipeline_name,
                total_waves=len(waves),
                total_nodes=len(graph.nodes),
            )
            await self._notify_observer(observer_manager, event)

            # Evaluate policy for pipeline start - policy can provide timeout
            policy_response = await self._evaluate_policy(policy_manager, event, context)
            self._check_policy_signal(policy_response, "Pipeline start")

            # Extract timeout from policy response (if provided)
            timeout = None
            if policy_response.data and isinstance(policy_response.data, dict):
                timeout = policy_response.data.get("timeout")

            # Track pipeline status for post-DAG hooks

            pipeline_status: PipelineStatus = PipelineStatus.SUCCESS
            pipeline_error: BaseException | None = None
            cancelled = False

            try:
                # Execute all waves using executor (always uses executor now)
                cancelled = await self._execute_with_executor(
                    waves=waves,
                    graph=graph,
                    node_results=node_results,
                    initial_input=initial_input,
                    context=context,
                    timeout=timeout,
                    validate=validate,
                    **kwargs,
                )
            except BaseException as e:
                # Track error for post-hooks (do NOT modify state in handler!)
                pipeline_error = e
                raise  # Re-raise immediately
            else:
                # Success path - determine status after execution
                if cancelled:
                    pipeline_status = PipelineStatus.CANCELLED
            finally:
                # Update status based on what happened (outside try/except)
                if pipeline_error is not None:
                    pipeline_status = PipelineStatus.FAILED
                # Fire appropriate completion/cancellation event
                duration_ms = (time.time() - pipeline_start_time) * 1000

                if cancelled:
                    pipeline_cancelled = PipelineCancelled(
                        name=pipeline_name,
                        duration_ms=duration_ms,
                        reason="timeout",
                        partial_results=node_results,
                    )
                    await self._notify_observer(observer_manager, pipeline_cancelled)
                elif pipeline_status == PipelineStatus.SUCCESS:
                    pipeline_completed = PipelineCompleted(
                        name=pipeline_name,
                        duration_ms=duration_ms,
                        node_results=node_results,
                    )
                    await self._notify_observer(observer_manager, pipeline_completed)

                # POST-DAG HOOKS: Always execute for cleanup (even on failure)
                try:
                    post_hook_results = await self._post_hook_manager.execute_hooks(
                        context=context,
                        pipeline_name=pipeline_name,
                        pipeline_status=pipeline_status.value,
                        node_results=node_results,
                        error=pipeline_error,
                    )
                    # Store post-hook results in context (for debugging/logging)
                    context.metadata["post_dag_hooks"] = post_hook_results
                except Exception as post_hook_error:
                    # Log all hook errors but don't fail the pipeline
                    # (hooks are for cleanup/observability, not critical path)
                    logger.error(
                        f"Post-DAG hooks failed: {post_hook_error}",
                        exc_info=True,
                    )

        # Return results (outside ExecutionContext - it's cleaned up automatically)
        return node_results

    async def _execute_with_executor(
        self,
        waves: list[list[str]],
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        context: NodeExecutionContext,
        timeout: float | None,
        validate: bool,
        **kwargs: Any,
    ) -> bool:
        """Execute all waves using the configured executor.

        This method delegates execution to the executor (LocalExecutor by default,
        or CeleryExecutor, AzureFunctionsExecutor, etc. if provided).

        Parameters
        ----------
        waves : list[list[str]]
            List of execution waves
        graph : DirectedGraph
            The DAG to execute
        node_results : dict[str, Any]
            Accumulated results from previous waves
        initial_input : Any
            Initial input to the pipeline
        context : NodeExecutionContext
            Execution context
        timeout : float | None
            Optional timeout for entire execution
        validate : bool
            Whether to perform validation
        **kwargs : Any
            Additional parameters

        Returns
        -------
        bool
            True if cancelled (timeout), False if completed
        """
        # Store graph, node_results, and initial_input in ports context for executor
        # Note: We extend the existing ports with executor-specific context.
        # This is safe because set_ports() wraps in MappingProxyType (immutable).
        existing_ports_result = get_ports()
        existing_ports: dict[str, Any] = (
            dict(existing_ports_result) if existing_ports_result else {}
        )
        executor_ports = {
            **existing_ports,
            EXECUTOR_CONTEXT_GRAPH: graph,
            EXECUTOR_CONTEXT_NODE_RESULTS: node_results,
            EXECUTOR_CONTEXT_INITIAL_INPUT: initial_input,
        }
        set_ports(executor_ports)

        try:
            async with asyncio.timeout(timeout):
                # Dynamic wave execution - re-compute waves if graph expands
                wave_idx = 0
                previous_wave_count = len(waves)

                while wave_idx < len(waves):
                    wave = waves[wave_idx]
                    wave_start_time = time.time()

                    # Fire wave started event
                    wave_event = WaveStarted(wave_index=wave_idx + 1, nodes=wave)
                    await self._notify_observer(get_observer_manager(), wave_event)

                    # Create tasks for this wave
                    tasks = []
                    for node_name in wave:
                        task = ExecutionTask(
                            node_name=node_name,
                            node_input=None,  # Executor will prepare input from graph
                            wave_index=wave_idx + 1,
                            should_validate=validate,
                            context_data={
                                "dag_id": context.dag_id,
                                "run_id": context.metadata.get("run_id"),
                                "attempt": context.attempt,
                            },
                            params=kwargs,
                        )
                        tasks.append(task)

                    # Execute wave using executor
                    wave_results = await self.executor.aexecute_wave(tasks)

                    # Update node_results with wave results
                    # Note: Failures propagate as NodeExecutionError from executor,
                    # so we only see SUCCESS results here
                    for node_name, result in wave_results.items():
                        node_results[node_name] = result.output

                    # Update node_results in context for expander nodes
                    set_node_results(node_results)

                    # Fire wave completed event
                    wave_completed = WaveCompleted(
                        wave_index=wave_idx + 1,
                        duration_ms=(time.time() - wave_start_time) * 1000,
                    )
                    await self._notify_observer(get_observer_manager(), wave_completed)

                    # Check if graph was dynamically expanded
                    new_waves = graph.waves()
                    if len(new_waves) != previous_wave_count:
                        # Graph expanded! Re-compute waves
                        logger.info(
                            "Dynamic expansion detected: {old} → {new} waves",
                            old=previous_wave_count,
                            new=len(new_waves),
                        )
                        waves = new_waves
                        previous_wave_count = len(new_waves)
                        # Don't increment wave_idx - continue from where we are

                    # Move to next wave
                    wave_idx += 1

            return False  # Not cancelled

        except TimeoutError:
            return True  # Cancelled due to timeout

    def _resolve_ports_for_node(self, node_name: str, node_spec: Any) -> dict[str, Any]:
        """Resolve ports for a specific node.

        Uses PortsConfiguration if available, otherwise returns global ports.
        Resolution order: per-node > per-type > global

        Args
        ----
            node_name: Name of the node
            node_spec: NodeSpec containing node metadata

        Returns
        -------
            dict[str, Any]: Resolved ports for this node
        """
        if self.ports_config is None:
            # No PortsConfiguration, use global ports
            return self.ports

        # Get node type from spec (if available)
        node_type = getattr(node_spec, "subtype", None)
        if node_type:
            node_type = node_type.value if hasattr(node_type, "value") else str(node_type)

        # Resolve ports using PortsConfiguration
        resolved_ports = self.ports_config.resolve_ports(node_name, node_type)

        # Convert PortConfig to actual port instances
        return {k: v.port for k, v in resolved_ports.items()}
