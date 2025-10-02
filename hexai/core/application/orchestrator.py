"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from hexai.core.ports.observer_manager import ObserverManagerPort
    from hexai.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexai.core.context import ExecutionContext, get_policy_manager
from hexai.core.domain.dag import DirectedGraph
from hexai.core.orchestration import NodeExecutionContext
from hexai.core.orchestration.components import (
    InputMapper,
    NodeExecutionError,
    NodeExecutor,
    OrchestratorError,
    PolicyCoordinator,
    WaveExecutor,
)
from hexai.core.orchestration.hooks import (
    HookConfig,
    PostDagHookConfig,
    PostDagHookManager,
    PreDagHookManager,
)

from .events import (
    NodeFailed,
    PipelineCancelled,
    PipelineCompleted,
    PipelineStarted,
)
from .policies.models import PolicyResponse, PolicySignal
from .ports_builder import PortsBuilder

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_MAX_CONCURRENT_NODES = 10


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
        ports: dict[str, Any] | None = None,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        pre_hook_config: HookConfig | None = None,
        post_hook_config: PostDagHookConfig | None = None,
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Shared ports/dependencies for all pipeline executions
            strict_validation: If True, raise errors on validation failure
            default_node_timeout: Default timeout in seconds for each node (None = no timeout)
            pre_hook_config: Configuration for pre-DAG hooks (health checks, secrets, etc.)
            post_hook_config: Configuration for post-DAG hooks (cleanup, checkpoints, etc.)
        """
        self.max_concurrent_nodes = max_concurrent_nodes
        self.ports = ports or {}
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

        # Validate known port types
        self._validate_port_types(self.ports)

        # Initialize collaborators (dependency injection)
        self._policy_coordinator = PolicyCoordinator()
        self._input_mapper = InputMapper()
        self._node_executor = NodeExecutor(
            strict_validation=strict_validation,
            default_node_timeout=default_node_timeout,
        )

        # Initialize hook managers

        self._pre_hook_manager = PreDagHookManager(pre_hook_config)
        self._post_hook_manager = PostDagHookManager(post_hook_config, self._pre_hook_manager)
        self._wave_executor = WaveExecutor(max_concurrent_nodes=max_concurrent_nodes)

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

        for node_name, node_spec in graph.nodes.items():
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
    ) -> "Orchestrator":
        """Create an Orchestrator using a PortsBuilder.

        This provides a more intuitive way to configure the orchestrator
        with type-safe port configuration.

        Args
        ----
            builder: Configured PortsBuilder instance
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            strict_validation: If True, raise errors on validation failure

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

        Raises
        ------
        OrchestratorError
            If DAG validation fails or pipeline/wave execution is blocked

        Examples
        --------
        Using dictionary for additional ports (traditional approach):

        >>> results = await orchestrator.run(
        ...     graph,
        ...     input_data,
        ...     additional_ports={"llm": MockLLM()}
        ... )

        Using PortsBuilder for additional ports (new approach):

        >>> results = await orchestrator.run(
        ...     graph,
        ...     input_data,
        ...     additional_ports=PortsBuilder().with_llm(MockLLM())
        ... )
        """
        # Merge orchestrator ports with additional execution-specific ports
        all_ports = {**self.ports}
        if additional_ports:
            # Handle both dictionary and PortsBuilder
            if isinstance(additional_ports, PortsBuilder):
                all_ports.update(additional_ports.build())
            else:
                all_ports.update(additional_ports)
        if validate:
            try:
                # By default, skip type checking for backward compatibility
                # Enable via graph.validate(check_type_compatibility=True)
                graph.validate(check_type_compatibility=False)
            except Exception as e:
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

            pipeline_status: Literal["success", "failed", "cancelled"] = "success"
            pipeline_error: Exception | None = None
            cancelled = False

            try:
                # Execute all waves with optional timeout (delegate to WaveExecutor)
                cancelled = await self._wave_executor.execute_all_waves(
                    waves=waves,
                    node_executor_fn=self._execute_node,
                    graph=graph,
                    node_results=node_results,
                    initial_input=initial_input,
                    all_ports=all_ports,
                    context=context,
                    observer_manager=observer_manager,
                    policy_manager=policy_manager,
                    policy_coordinator=self._policy_coordinator,
                    timeout=timeout,
                    validate=validate,
                    **kwargs,
                )

                # Determine final status
                if cancelled:
                    pipeline_status = "cancelled"

            except Exception as e:
                pipeline_status = "failed"
                pipeline_error = e
                raise  # Re-raise to preserve stack trace

            finally:
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
                elif pipeline_status == "success":
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
                        pipeline_status=pipeline_status,
                        node_results=node_results,
                        error=pipeline_error,
                    )
                    # Store post-hook results in context (for debugging/logging)
                    context.metadata["post_dag_hooks"] = post_hook_results

                except Exception as post_hook_error:
                    # Log but don't fail the pipeline due to post-hook errors
                    logger.error(
                        f"Post-DAG hooks failed: {post_hook_error}",
                        exc_info=True,
                    )

        # Return results (outside ExecutionContext - it's cleaned up automatically)
        return node_results

    async def _execute_all_waves(
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
        """Execute all waves with optional timeout (delegates to WaveExecutor)."""
        return await self._wave_executor.execute_all_waves(
            waves=waves,
            node_executor_fn=self._execute_node,
            graph=graph,
            node_results=node_results,
            initial_input=initial_input,
            context=context,
            policy_coordinator=self._policy_coordinator,
            timeout=timeout,
            validate=validate,
            **kwargs,
        )

    async def _execute_node(
        self,
        node_name: str,
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        context: NodeExecutionContext,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node (delegates to NodeExecutor with retry logic).

        Note: all_ports, observer_manager, policy_manager are filtered out from kwargs
        since they're accessed via ExecutionContext, not passed to node functions.
        """
        # Create node context early so it's available in exception handler
        _ = context.with_node(node_name, wave_index)  # For future use

        # Filter out orchestrator-level kwargs that shouldn't be passed to nodes
        # These are available via ExecutionContext instead
        node_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"all_ports", "observer_manager", "policy_manager"}
        }

        try:
            # Prepare input using InputMapper
            node_spec = graph.nodes[node_name]
            node_input = self._input_mapper.prepare_node_input(
                node_spec, node_results, initial_input
            )

            # Delegate execution to NodeExecutor
            return await self._node_executor.execute_node(
                node_name=node_name,
                node_spec=node_spec,
                node_input=node_input,
                context=context,
                policy_coordinator=self._policy_coordinator,
                wave_index=wave_index,
                validate=validate,
                **node_kwargs,
            )

        except NodeExecutionError as e:
            # Check if this is a RETRY signal from the policy
            # The orchestrator handles retries at this level
            if hasattr(e, "__cause__") and isinstance(e.__cause__, Exception):
                # Try to get retry signal from policy
                fail_event = NodeFailed(
                    name=node_name,
                    wave_index=wave_index,
                    error=e.__cause__,
                )
                fail_response = await self._evaluate_policy(
                    get_policy_manager(),
                    fail_event,
                    context,
                    node_id=node_name,
                    wave_index=wave_index,
                )

                if fail_response.signal == PolicySignal.FALLBACK:
                    return fail_response.data
                if fail_response.signal == PolicySignal.RETRY:
                    retry_data = fail_response.data if isinstance(fail_response.data, dict) else {}
                    delay = retry_data.get("delay", 0)

                    if delay > 0:
                        await asyncio.sleep(delay)

                    # Retry
                    return await self._execute_node(
                        node_name=node_name,
                        graph=graph,
                        node_results=node_results,
                        initial_input=initial_input,
                        context=context.with_attempt(context.attempt + 1),
                        wave_index=wave_index,
                        validate=validate,
                        **kwargs,
                    )
            # Default: propagate
            raise
