"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.ports.observer_manager import ObserverManagerPort
    from hexai.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexai.core.domain.dag import DirectedGraph, NodeSpec, ValidationError
from hexai.core.orchestration import NodeExecutionContext

from .events import (
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCancelled,
    PipelineCompleted,
    PipelineStarted,
)
from .policies.models import PolicyContext, PolicyResponse, PolicySignal
from .ports_builder import PortsBuilder

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_MAX_CONCURRENT_NODES = 10


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class NodeExecutionError(OrchestratorError):
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


class PolicyCoordinator:
    """Handles policy evaluation and observer notifications.

    Single responsibility: Coordinate policy decisions and event observation.
    """

    async def notify_observer(
        self, observer_manager: ObserverManagerPort | None, event: Any
    ) -> None:
        """Notify observer if it exists.

        Parameters
        ----------
        observer_manager : ObserverManagerPort | None
            Observer manager to notify
        event : Any
            Event to send
        """
        if observer_manager:
            await observer_manager.notify(event)

    async def evaluate_policy(
        self,
        policy_manager: PolicyManagerPort | None,
        event: Any,
        context: NodeExecutionContext,
        node_id: str | None = None,
        wave_index: int | None = None,
        attempt: int = 1,
    ) -> PolicyResponse:
        """Evaluate policy and create context.

        Parameters
        ----------
        policy_manager : PolicyManagerPort | None
            Policy manager to evaluate
        event : Any
            Event triggering policy evaluation
        context : ExecutionContext
            Current execution context
        node_id : str | None
            Optional node ID override
        wave_index : int | None
            Optional wave index override
        attempt : int
            Attempt number for retries

        Returns
        -------
        PolicyResponse
            Policy decision
        """
        policy_context = PolicyContext(
            event=event,
            dag_id=context.dag_id,
            node_id=node_id or context.node_id,
            wave_index=wave_index or context.wave_index,
            attempt=attempt or context.attempt,
        )

        if policy_manager:
            return await policy_manager.evaluate(policy_context)
        return PolicyResponse()  # Default: proceed

    def check_policy_signal(self, response: PolicyResponse, context: str) -> None:
        """Check policy signal and raise error if not PROCEED.

        Parameters
        ----------
        response : PolicyResponse
            Policy response to check
        context : str
            Context description for error message

        Raises
        ------
        OrchestratorError
            If policy signal is not PROCEED
        """
        if response.signal != PolicySignal.PROCEED:
            raise OrchestratorError(f"{context} blocked: {response.signal.value}")


class InputMapper:
    """Handles input data preparation and mapping for nodes.

    Single responsibility: Map dependencies to node inputs.
    """

    def prepare_node_input(
        self, node_spec: NodeSpec, node_results: dict[str, Any], initial_input: Any
    ) -> Any:
        """Prepare input data for node execution with simplified data mapping.

        Parameters
        ----------
        node_spec : NodeSpec
            Node specification
        node_results : dict[str, Any]
            Results from previously executed nodes
        initial_input : Any
            Initial input data for the pipeline

        Returns
        -------
        Any
            Prepared input data for the node
        """
        if not node_spec.deps:
            # No dependencies - use initial input
            return initial_input

        if len(node_spec.deps) == 1:
            # Single dependency - pass through directly
            dep_name = next(iter(node_spec.deps))
            return node_results.get(dep_name, initial_input)

        # Multiple dependencies - preserve namespace structure
        aggregated_data = {}

        # Keep dependency results with their node names as keys
        for dep_name in node_spec.deps:
            if dep_name in node_results:
                aggregated_data[dep_name] = node_results[dep_name]

        return aggregated_data


class NodeExecutor:
    """Handles individual node execution with validation, timeout, and retry logic.

    Single responsibility: Execute a single node with all its lifecycle concerns.
    """

    def __init__(
        self,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
    ) -> None:
        """Initialize node executor.

        Parameters
        ----------
        strict_validation : bool
            If True, raise errors on validation failure
        default_node_timeout : float | None
            Default timeout in seconds for each node
        """
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

    async def execute_node(
        self,
        node_name: str,
        node_spec: NodeSpec,
        node_input: Any,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        policy_coordinator: PolicyCoordinator,
        observer_manager: ObserverManagerPort | None,
        policy_manager: PolicyManagerPort | None,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node with full lifecycle management.

        Parameters
        ----------
        node_name : str
            Name of the node
        node_spec : NodeSpec
            Node specification
        node_input : Any
            Prepared input data
        ports : dict[str, Any]
            Available ports
        context : ExecutionContext
            Execution context
        policy_coordinator : PolicyCoordinator
            Policy coordinator instance
        observer_manager : ObserverManagerPort | None
            Observer manager
        policy_manager : PolicyManagerPort | None
            Policy manager
        wave_index : int
            Wave index
        validate : bool
            Whether to validate inputs/outputs
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        Any
            Output of the node

        Raises
        ------
        NodeExecutionError
            If the node fails to execute
        NodeTimeoutError
            If node execution exceeds timeout
        OrchestratorError
            If the node is blocked
        ValidationError
            If the node input is invalid
        """
        node_start_time = time.time()

        try:
            # Input validation
            if validate:
                try:
                    validated_input = node_spec.validate_input(node_input)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    logger.debug("Input validation failed for node '%s': %s", node_name, e)
                    validated_input = node_input
            else:
                validated_input = node_input

            # Fire node started event and check policy
            start_event = NodeStarted(
                name=node_name,
                wave_index=wave_index,
                dependencies=list(node_spec.deps),
            )
            await policy_coordinator.notify_observer(observer_manager, start_event)

            # Evaluate policy for node start
            start_response = await policy_coordinator.evaluate_policy(
                policy_manager, start_event, context, node_id=node_name, wave_index=wave_index
            )

            # Handle control signals
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

            try:
                if node_timeout:
                    async with asyncio.timeout(node_timeout):
                        raw_output = await self._execute_function(
                            node_spec, validated_input, ports, kwargs
                        )
                else:
                    raw_output = await self._execute_function(
                        node_spec, validated_input, ports, kwargs
                    )
            except TimeoutError as e:
                # node_timeout is guaranteed to be set here because TimeoutError
                # only occurs when timeout is set
                timeout_value = node_timeout if node_timeout is not None else 0.0
                raise NodeTimeoutError(node_name, timeout_value, e) from e

            # Output validation
            if validate:
                try:
                    validated_output = node_spec.validate_output(raw_output)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    logger.debug("Output validation failed for node '%s': %s", node_name, e)
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
            await policy_coordinator.notify_observer(observer_manager, complete_event)

            return validated_output

        except NodeTimeoutError:
            # Node timed out - emit cancelled event and re-raise
            cancel_event = NodeCancelled(
                name=node_name,
                wave_index=wave_index,
                reason="timeout",
            )
            await policy_coordinator.notify_observer(observer_manager, cancel_event)
            raise

        except Exception as e:
            # Fire node failed event and check policy
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=e,
            )
            await policy_coordinator.notify_observer(observer_manager, fail_event)

            # Evaluate policy for node failure
            fail_response = await policy_coordinator.evaluate_policy(
                policy_manager, fail_event, context, node_id=node_name, wave_index=wave_index
            )

            # Handle retry signal
            if fail_response.signal == PolicySignal.RETRY:
                # Re-raise to let orchestrator handle retry
                raise

            # Default: propagate the error
            raise NodeExecutionError(node_name, e) from e

    async def _execute_function(
        self,
        node_spec: NodeSpec,
        validated_input: Any,
        ports: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute node function with proper async/sync handling.

        Parameters
        ----------
        node_spec : NodeSpec
            The node specification containing the function
        validated_input : Any
            Validated input data
        ports : dict[str, Any]
            Available ports
        kwargs : dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        Any
            Output from the node function
        """
        if asyncio.iscoroutinefunction(node_spec.fn):
            return await node_spec.fn(validated_input, **ports, **kwargs)
        # Run sync functions in executor to avoid blocking event loop
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: node_spec.fn(validated_input, **ports, **kwargs)
        )


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
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Shared ports/dependencies for all pipeline executions
            strict_validation: If True, raise errors on validation failure
            default_node_timeout: Default timeout in seconds for each node (None = no timeout)
        """
        self.max_concurrent_nodes = max_concurrent_nodes
        self.ports = ports or {}
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

        # Initialize collaborators (dependency injection)
        self._policy_coordinator = PolicyCoordinator()
        self._input_mapper = InputMapper()
        self._node_executor = NodeExecutor(
            strict_validation=strict_validation,
            default_node_timeout=default_node_timeout,
        )

        # Import and initialize WaveExecutor
        from hexai.core.orchestration.components import WaveExecutor

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

        # Execute all waves with optional timeout
        cancelled = await self._execute_all_waves(
            waves=waves,
            graph=graph,
            node_results=node_results,
            initial_input=initial_input,
            all_ports=all_ports,
            context=context,
            observer_manager=observer_manager,
            policy_manager=policy_manager,
            timeout=timeout,
            validate=validate,
            **kwargs,
        )

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
        else:
            pipeline_completed = PipelineCompleted(
                name=pipeline_name,
                duration_ms=duration_ms,
                node_results=node_results,
            )
            await self._notify_observer(observer_manager, pipeline_completed)

        return node_results

    async def _execute_all_waves(
        self,
        waves: list[list[str]],
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        all_ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        policy_manager: PolicyManagerPort | None,
        timeout: float | None,
        validate: bool,
        **kwargs: Any,
    ) -> bool:
        """Execute all waves with optional timeout (delegates to WaveExecutor).

        Args
        ----
            waves: List of waves to execute
            graph: The DirectedGraph being executed
            node_results: Dictionary to accumulate node results
            initial_input: Initial input data
            all_ports: All available ports
            context: Execution context
            observer_manager: Optional observer manager
            policy_manager: Optional policy manager
            timeout: Optional timeout in seconds
            validate: Whether to validate nodes
            **kwargs: Additional arguments

        Returns
        -------
        bool
            True if execution was cancelled due to timeout, False otherwise
        """
        return await self._wave_executor.execute_all_waves(
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

    async def _execute_node(
        self,
        node_name: str,
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        policy_manager: PolicyManagerPort | None,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node (delegates to NodeExecutor with retry logic).

        The orchestrator provides the MECHANISM for retries through the RETRY signal,
        but the retry POLICY (when to retry, how many times, delays) is determined
        by control handlers registered with the EventBus.

        Returns
        -------
            Output of the node

        Raises
        ------
        NodeExecutionError
            If the node fails to execute
        """
        # Create node context early so it's available in exception handler
        _ = context.with_node(node_name, wave_index)  # For future use

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
                ports=ports,
                context=context,
                policy_coordinator=self._policy_coordinator,
                observer_manager=observer_manager,
                policy_manager=policy_manager,
                wave_index=wave_index,
                validate=validate,
                **kwargs,
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
                    policy_manager, fail_event, context, node_id=node_name, wave_index=wave_index
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
                        ports=ports,
                        context=context.with_attempt(context.attempt + 1),
                        observer_manager=observer_manager,
                        policy_manager=policy_manager,
                        wave_index=wave_index,
                        validate=validate,
                        **kwargs,
                    )
            # Default: propagate
            raise
