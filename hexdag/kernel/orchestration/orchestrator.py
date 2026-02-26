"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel.ports.observer_manager import ObserverManager
else:
    ObserverManager = Any

from hexdag.kernel.context import (
    ExecutionContext,
    get_observer_manager,
    get_ports,
    set_current_graph,
    set_node_results,
    set_ports,
    set_ports_config,
    set_services,
)
from hexdag.kernel.domain.dag import DirectedGraph, DirectedGraphError
from hexdag.kernel.exceptions import OrchestratorError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.components import ExecutionCoordinator
from hexdag.kernel.orchestration.components.lifecycle_manager import (
    HookConfig,
    LifecycleManager,
    PostDagHookConfig,
)
from hexdag.kernel.orchestration.constants import (
    EXECUTOR_CONTEXT_GRAPH,
    EXECUTOR_CONTEXT_INITIAL_INPUT,
    EXECUTOR_CONTEXT_NODE_RESULTS,
)
from hexdag.kernel.orchestration.events import WaveCompleted
from hexdag.kernel.orchestration.hook_context import PipelineStatus
from hexdag.kernel.orchestration.models import NodeExecutionContext, PortsConfiguration
from hexdag.kernel.orchestration.port_wrappers import wrap_ports_with_observability
from hexdag.kernel.ports.executor import ExecutionResult, ExecutionTask, Executor
from hexdag.kernel.ports_builder import PortsBuilder
from hexdag.kernel.utils.node_timer import Timer

from .events import PipelineCompleted, PipelineStarted

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
    executor: Executor | None = None,
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
    executor : Executor | None
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
            logger.error("Executor setup failed: {}", e)
            raise

    # Setup ports
    initialized: list[str] = []
    for name, port in all_ports.items():
        if _has_async_lifecycle(port, "asetup"):
            try:
                await port.asetup()
                initialized.append(name)
            except Exception as e:
                logger.error("Port setup failed: {}: {}", name, e)
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
                    logger.warning("Port cleanup failed: {}: {}", name, e)

        # Cleanup executor last
        if (
            executor_initialized
            and executor is not None
            and _has_async_lifecycle(executor, "aclose")
        ):
            try:
                await executor.aclose()
            except Exception as e:
                logger.warning("Executor cleanup failed: {}", e)


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
        ports: dict[str, Any] | PortsConfiguration | None = None,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        pre_hook_config: HookConfig | None = None,
        post_hook_config: PostDagHookConfig | None = None,
        executor: Executor | None = None,
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
                Set to a custom Executor implementation (e.g., CeleryExecutor,
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
            from hexdag.drivers.executors import (
                LocalExecutor,  # noqa: PLC0415  # lazy: deferred to avoid loading executor at class import
            )

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

        self.ports_config: PortsConfiguration | None
        if isinstance(ports, PortsConfiguration):
            self.ports_config = ports
            # Note: global_ports is stored as tuple after __post_init__, convert to dict
            self.ports = (
                {k: v.port for k, v in dict(ports.global_ports).items()}
                if ports.global_ports
                else {}
            )
        else:
            self.ports_config = None
            self.ports = ports or {}

        # Validate known port types
        self._validate_port_types(self.ports)

        # Unified managers (consolidate 11 managers into 2 unified managers)
        self._execution_coordinator = ExecutionCoordinator()
        self._lifecycle_manager = LifecycleManager(pre_hook_config, post_hook_config)

    async def _notify_observer(self, observer_manager: ObserverManager | None, event: Any) -> None:
        """Notify observer if it exists (delegates to ExecutionCoordinator)."""
        await self._execution_coordinator.notify_observer(observer_manager, event)

    def _validate_port_types(self, ports: dict[str, Any]) -> None:
        """Validate that orchestrator ports match expected types.

        Args
        ----
            ports: Dictionary of ports to validate

        Notes
        -----
        This provides helpful warnings if ports don't match expected protocols.
        Currently checks observer_manager.
        """
        # Check observer_manager if provided
        if "observer_manager" in ports:
            obs = ports["observer_manager"]
            if not hasattr(obs, "notify"):
                logger.warning(
                    "Port 'observer_manager' doesn't have 'notify' method. "
                    "Expected ObserverManager, got {}",
                    type(obs).__name__,
                )

    def _validate_port_capabilities(
        self, graph: DirectedGraph, available_ports: dict[str, Any]
    ) -> None:
        """Validate port existence and capabilities at mount time.

        Like Linux VFS checks ``f_op`` at ``open()`` — not at ``read()`` time.
        The capability dict keys define required ports; the values define
        required ``Supports*`` protocols.

        Parameters
        ----------
        graph : DirectedGraph
            The DAG to validate.
        available_ports : dict[str, Any]
            Dictionary of available port adapters.

        Raises
        ------
        OrchestratorError
            If any required port is missing or lacks a required capability.
        """
        errors: list[str] = []

        for node_name, node_spec in graph.items():
            fn = node_spec.fn
            capabilities: dict[str, list[type]] = {}

            if hasattr(fn, "_hexdag_port_capabilities"):
                capabilities = getattr(fn, "_hexdag_port_capabilities", {})
            elif (self_obj := getattr(fn, "__self__", None)) is not None:
                capabilities = getattr(self_obj.__class__, "_hexdag_port_capabilities", {})

            for port_name, protocols in capabilities.items():
                port = available_ports.get(port_name)
                if port is None:
                    errors.append(f"  Node '{node_name}': missing required port '{port_name}'")
                    continue

                errors.extend(
                    f"  Node '{node_name}': port '{port_name}' "
                    f"({type(port).__name__}) does not implement "
                    f"{protocol.__name__}"
                    for protocol in protocols
                    if not isinstance(port, protocol)
                )

        if errors:
            raise OrchestratorError("Port capability validation failed:\n" + "\n".join(errors))

    def _check_wave_results_for_failures(self, wave_results: dict[str, ExecutionResult]) -> None:
        """Check wave results for failures and raise appropriate errors.

        Parameters
        ----------
        wave_results : dict[str, ExecutionResult]
            Results from executing a wave of nodes

        Raises
        ------
        OrchestratorError
            If any node in the wave failed
        """
        failed_nodes = [
            (name, result)
            for name, result in wave_results.items()
            if result.status == PipelineStatus.FAILED
        ]

        if failed_nodes:
            # Collect all failures into one error message
            error_parts = []
            for node_name, result in failed_nodes:
                error_type = result.error_type or "Unknown"
                error_msg = result.error or "No error message"
                error_parts.append(f"  - Node '{node_name}' ({error_type}): {error_msg}")

            raise OrchestratorError(
                f"Pipeline execution failed. {len(failed_nodes)} node(s) failed:\n"
                + "\n".join(error_parts)
            )

    @classmethod
    def from_builder(
        cls,
        builder: PortsBuilder,
        max_concurrent_nodes: int = DEFAULT_MAX_CONCURRENT_NODES,
        strict_validation: bool = False,
        executor: Executor | None = None,
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
        dynamic: bool = False,
        max_dynamic_iterations: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a DAG with concurrent processing and resource limits.

        Supports both traditional dictionary-based ports and the new PortsBuilder
        for additional_ports parameter. When using PortsBuilder, it will be
        automatically converted to a dictionary before merging with base ports.

        Args
        ----
            graph: The DirectedGraph to execute
            initial_input: Initial input data for the graph
            additional_ports: Either a dictionary of ports or a PortsBuilder instance
            validate: Whether to validate the graph before execution
            dynamic: Enable dynamic graph expansion (for agent macros).
                When True, supports:
                - Runtime node injection via get_current_graph()
                - Re-execution of nodes that return None
                - Iterative expansion until all nodes complete
            max_dynamic_iterations: Maximum number of expansion iterations (safety limit).
                Prevents infinite loops in dynamic execution.
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
                graph, initial_input, all_ports, validate, dynamic, max_dynamic_iterations, **kwargs
            )

    async def _execute_with_ports(
        self,
        graph: DirectedGraph,
        initial_input: Any,
        all_ports: dict[str, Any],
        validate: bool,
        dynamic: bool,
        max_dynamic_iterations: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute DAG with managed ports (internal method).

        This method is separated to work with the managed_ports context manager.

        Args
        ----
            graph: DirectedGraph to execute
            initial_input: Initial input data
            all_ports: Merged ports dictionary
            validate: Whether to validate graph
            dynamic: Enable dynamic graph expansion
            max_dynamic_iterations: Maximum dynamic expansion iterations
            **kwargs: Additional arguments
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

        # Validate port existence and capabilities at mount time
        self._validate_port_capabilities(graph, all_ports)

        node_results: dict[str, Any] = {}
        waves = graph.waves()
        pipeline_timer = Timer()

        observer_manager: ObserverManager | None = all_ports.get("observer_manager")

        wrapped_ports = wrap_ports_with_observability(all_ports)

        pipeline_name = getattr(graph, "name", "unnamed")
        context = NodeExecutionContext(dag_id=pipeline_name)
        run_id = str(uuid.uuid4())

        async with ExecutionContext(
            observer_manager=observer_manager,
            run_id=run_id,
            ports=wrapped_ports,
        ):
            set_current_graph(graph)
            set_node_results(node_results)
            # Set PortsConfiguration for per-node port resolution
            if self.ports_config is not None:
                set_ports_config(self.ports_config)

            # Set services in context for agent tool injection
            services = wrapped_ports.get("_hexdag_services")
            if services:
                set_services(services)

            # PRE-DAG LIFECYCLE: Execute before pipeline starts
            pre_hook_results = await self._lifecycle_manager.pre_execute(
                context=context,
                pipeline_name=pipeline_name,
            )
            context.metadata["pre_dag_hooks"] = pre_hook_results

            # Fire pipeline started event
            event = PipelineStarted(
                name=pipeline_name,
                total_waves=len(waves),
                total_nodes=len(graph),
            )
            await self._notify_observer(observer_manager, event)

            timeout = None
            pipeline_status: PipelineStatus = PipelineStatus.SUCCESS
            pipeline_error: BaseException | None = None
            cancelled = False

            try:
                # Route to appropriate execution mode
                if dynamic:
                    # Dynamic execution with runtime node injection
                    cancelled = await self._execute_dynamic(
                        graph=graph,
                        node_results=node_results,
                        initial_input=initial_input,
                        context=context,
                        timeout=timeout,
                        validate=validate,
                        max_iterations=max_dynamic_iterations,
                        **kwargs,
                    )
                else:
                    # Static execution (traditional wave-based)
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
                pipeline_error = e
                raise  # Re-raise immediately
            else:
                # Success path - determine status after execution
                if cancelled:
                    pipeline_status = PipelineStatus.CANCELLED
            finally:
                if pipeline_error is not None:
                    pipeline_status = PipelineStatus.FAILED
                # Fire appropriate completion/cancellation event
                duration_ms = pipeline_timer.duration_ms

                if cancelled:
                    pipeline_completed = PipelineCompleted(
                        name=pipeline_name,
                        duration_ms=duration_ms,
                        node_results=node_results,
                        status="cancelled",
                        reason="timeout",
                    )
                    await self._notify_observer(observer_manager, pipeline_completed)
                elif pipeline_status == PipelineStatus.SUCCESS:
                    pipeline_completed = PipelineCompleted(
                        name=pipeline_name,
                        duration_ms=duration_ms,
                        node_results=node_results,
                    )
                    await self._notify_observer(observer_manager, pipeline_completed)

                # POST-DAG LIFECYCLE: Always execute for cleanup (even on failure)
                try:
                    post_hook_results = await self._lifecycle_manager.post_execute(
                        context=context,
                        pipeline_name=pipeline_name,
                        pipeline_status=pipeline_status.value,
                        node_results=node_results,
                        error=pipeline_error,
                    )
                    context.metadata["post_dag_hooks"] = post_hook_results
                except Exception as post_hook_error:
                    # Log all hook errors but don't fail the pipeline
                    # (hooks are for cleanup/observability, not critical path)
                    logger.error(
                        "Post-DAG lifecycle failed: {}",
                        post_hook_error,
                        exc_info=True,
                    )

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
                    wave_timer = Timer()

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

                    # Check for failed nodes and raise errors
                    self._check_wave_results_for_failures(wave_results)

                    # Store successful results
                    for node_name, result in wave_results.items():
                        node_results[node_name] = result.output

                    set_node_results(node_results)

                    # Fire wave completed event
                    wave_completed = WaveCompleted(
                        wave_index=wave_idx + 1,
                        duration_ms=wave_timer.duration_ms,
                        nodes=list(wave),
                    )
                    await self._notify_observer(get_observer_manager(), wave_completed)

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

    async def _execute_dynamic(
        self,
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        context: NodeExecutionContext,
        timeout: float | None,
        validate: bool,
        max_iterations: int,
        **kwargs: Any,
    ) -> bool:
        """Execute graph with dynamic node injection support.

        This method supports:
        1. Runtime node injection via get_current_graph()
        2. Re-execution of nodes that return None
        3. Iterative expansion until completion

        Parameters
        ----------
        graph : DirectedGraph
            The graph being executed (may be modified at runtime)
        node_results : dict[str, Any]
            Dictionary to store node execution results
        initial_input : Any
            Initial input data for the pipeline
        context : NodeExecutionContext
            Execution context for the pipeline
        timeout : float | None
            Optional timeout for the entire execution
        validate : bool
            Whether to validate nodes
        max_iterations : int
            Maximum number of dynamic expansion iterations
        **kwargs : Any
            Additional arguments

        Returns
        -------
        bool
            True if execution was cancelled, False otherwise

        Raises
        ------
        OrchestratorError
            If max_iterations is exceeded (infinite loop protection)
        """
        executed_nodes: set[str] = set()
        iteration = 0
        dynamic_timer = Timer()

        logger.info("Starting dynamic execution (max_iterations={})", max_iterations)

        while iteration < max_iterations:
            iteration += 1

            # Check timeout
            if timeout and dynamic_timer.duration_ms > timeout * 1000:
                logger.warning("Dynamic execution timeout reached")
                return True  # Cancelled

            # Get current graph state (may have new nodes injected)
            current_node_names = set(graph.keys())

            # Find nodes ready to execute
            ready_nodes = self._get_ready_nodes(
                graph=graph,
                all_node_names=current_node_names,
                executed_nodes=executed_nodes,
                node_results=node_results,
            )

            if not ready_nodes:
                # No more nodes to execute - we're done
                logger.info(
                    "Dynamic execution completed after {} iterations ({} nodes executed)",
                    iteration,
                    len(executed_nodes),
                )
                break

            logger.debug(
                "Dynamic iteration {}: executing {} nodes: {}",
                iteration,
                len(ready_nodes),
                ready_nodes,
            )

            # Execute wave of ready nodes
            wave_cancelled = await self._execute_with_executor(
                waves=[ready_nodes],  # Single wave
                graph=graph,
                node_results=node_results,
                initial_input=initial_input,
                context=context,
                timeout=timeout,
                validate=validate,
                **kwargs,
            )

            if wave_cancelled:
                return True  # Propagate cancellation

            # Mark nodes as executed
            # BUT: don't mark nodes that returned None (they need re-execution)
            for node_name in ready_nodes:
                result = node_results.get(node_name)
                if result is not None:
                    executed_nodes.add(node_name)
                    logger.debug("Node {} completed successfully", node_name)
                else:
                    logger.debug(
                        "Node {} returned None, will re-execute after dependencies",
                        node_name,
                    )

            # Update context for expander nodes
            set_node_results(node_results)

            # Check if new nodes were added to graph
            new_nodes = current_node_names.symmetric_difference(set(graph.keys()))
            if new_nodes:
                logger.info("Detected {} newly injected nodes: {}", len(new_nodes), new_nodes)

        # Check if we exceeded max iterations
        if iteration >= max_iterations:
            unexecuted = set(graph.keys()) - executed_nodes
            raise OrchestratorError(
                f"Dynamic execution exceeded max_iterations={max_iterations}. "
                f"Possible infinite loop. "
                f"Executed {len(executed_nodes)} nodes, "
                f"{len(unexecuted)} nodes remain: {unexecuted}"
            )

        return False  # Not cancelled

    def _get_ready_nodes(
        self,
        graph: DirectedGraph,
        all_node_names: set[str],
        executed_nodes: set[str],
        node_results: dict[str, Any],
    ) -> list[str]:
        """Get nodes that are ready to execute in dynamic mode.

        A node is ready if:
        1. It hasn't been executed successfully yet (not in executed_nodes)
        2. All its dependencies have completed (results available)

        Parameters
        ----------
        graph : DirectedGraph
            The current graph
        all_node_names : set[str]
            All node names in the current graph
        executed_nodes : set[str]
            Names of nodes that have been executed successfully
        node_results : dict[str, Any]
            Current execution results

        Returns
        -------
        list[str]
            List of node names ready to execute
        """
        ready = []

        for node_name in all_node_names:
            # Skip if already executed successfully
            if node_name in executed_nodes:
                continue

            node_spec = graph[node_name]

            # Check if all dependencies are satisfied
            deps_satisfied = all(dep in node_results for dep in node_spec.deps)

            if deps_satisfied:
                ready.append(node_name)

        return ready

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

        if node_type := getattr(node_spec, "subtype", None):
            node_type = node_type.value if hasattr(node_type, "value") else str(node_type)

        # Resolve ports using PortsConfiguration
        resolved_ports = self.ports_config.resolve_ports(node_name, node_type)

        return {k: v.port for k, v in resolved_ports.items()}
