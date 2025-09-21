"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import inspect
import logging
import time
from typing import Any

from hexai.core.application.policies.models import PolicyContext, PolicyResponse, PolicySignal
from hexai.core.application.ports_builder import PortsBuilder
from hexai.core.domain.dag import DirectedGraph, NodeSpec, ValidationError
from hexai.core.ports.policy_manager import PolicyManagerPort

from .events import (
    ExecutionContext,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    ObserverManager,
    PipelineCompleted,
    PipelineStarted,
    WaveCompleted,
    WaveStarted,
)

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class NodeExecutionError(OrchestratorError):
    """Exception raised when a node fails to execute."""

    def __init__(self, node_name: str, original_error: Exception) -> None:
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")


class PolicyEvaluationError(OrchestratorError):
    """Exception raised when policy evaluation fails."""

    def __init__(self, context: str, error: Exception) -> None:
        self.context = context
        self.error = error
        super().__init__(f"Policy evaluation failed at {context}: {error}")


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
        max_concurrent_nodes: int = 10,
        ports: "dict[str, Any] | PortsBuilder | None" = None,
        strict_validation: bool = False,
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Can be:
                - A PortsBuilder instance (recommended for new code)
                - A dictionary of ports (backward compatible)
                - None (will create a PortsBuilder with defaults)
            strict_validation: If True, raise errors on validation failure

        Examples
        --------
            # Using PortsBuilder (recommended)
            orchestrator = Orchestrator(
                ports=PortsBuilder()
                    .with_llm(MockLLM())
                    .with_database(PostgresAdapter())
                    .with_defaults()
            )

            # Backward compatible dictionary
            orchestrator = Orchestrator(
                ports={"llm": MockLLM(), "database": PostgresAdapter()}
            )

            # Auto-configure with defaults
            orchestrator = Orchestrator()  # Gets default managers
        """
        self.max_concurrent_nodes = max_concurrent_nodes
        self._semaphore = asyncio.Semaphore(max_concurrent_nodes)

        # Handle different port input types
        self.ports: dict[str, Any]
        if ports is None:
            # Auto-configure with defaults for convenience
            self.ports = PortsBuilder().with_defaults().build()
        elif isinstance(ports, PortsBuilder):
            # It's a PortsBuilder instance
            self.ports = ports.build()
        else:
            # It's already a dictionary (backward compatible)
            self.ports = ports

        self.strict_validation = strict_validation

    async def _evaluate_policy_safe(
        self, policy_manager: PolicyManagerPort, policy_context: PolicyContext, check_point: str
    ) -> PolicyResponse:
        """Safely evaluate policy with error handling and validation.

        Args:
            policy_manager: Policy manager to use
            policy_context: Context for policy evaluation
            check_point: Description of where evaluation is happening

        Returns:
            PolicyResponse from evaluation

        Raises:
            PolicyEvaluationError: If evaluation fails or returns invalid response
        """
        try:
            response = await policy_manager.evaluate(policy_context)

            # Validate response
            if response is None:
                raise PolicyEvaluationError(
                    check_point, ValueError("Policy manager returned None response")
                )

            if not hasattr(response, "signal"):
                raise PolicyEvaluationError(
                    check_point, ValueError("Invalid policy response: missing 'signal' attribute")
                )

            # Validate signal is a valid PolicySignal
            if not isinstance(response.signal, PolicySignal):
                raise PolicyEvaluationError(
                    check_point, ValueError(f"Invalid signal type: {type(response.signal)}")
                )

            return response

        except PolicyEvaluationError:
            raise
        except Exception as e:
            raise PolicyEvaluationError(check_point, e) from e

    def _convert_to_policy_context(
        self, event: Any, context: ExecutionContext, node_name: str | None = None
    ) -> PolicyContext:
        """Convert ExecutionContext and event to PolicyContext.

        Args:
            event: Event that triggered the policy check
            context: Current execution context
            node_name: Optional node name being executed

        Returns:
            PolicyContext for policy evaluation
        """
        # Extract error from event if it's a failure event
        error = None
        if hasattr(event, "error"):
            error = event.error

        # Build metadata from event and context
        metadata = {
            "event_type": type(event).__name__,
            "wave_index": getattr(event, "wave_index", None),
            "dag_id": context.dag_id,
            "attempt": context.attempt,
        }

        if hasattr(event, "duration_ms"):
            metadata["duration_ms"] = event.duration_ms

        return PolicyContext(
            dag_id=context.dag_id,
            node_id=node_name or context.node_id,
            attempt=context.attempt,
            error=error,
            metadata=metadata,
        )

    async def run(
        self,
        graph: DirectedGraph,
        initial_input: Any,
        additional_ports: dict[str, Any] | None = None,
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a DAG with concurrent processing and resource limits."""
        # Merge orchestrator ports with additional execution-specific ports
        all_ports = dict(self.ports) if self.ports else {}
        if additional_ports:
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

        # Get observer manager and policy manager from ports
        observer_manager: ObserverManager = all_ports.get("observer_manager", ObserverManager())

        # Get policy_manager from ports - required
        policy_manager: PolicyManagerPort | None = all_ports.get("policy_manager")
        if policy_manager is None:
            raise ValueError(
                "PolicyManager not found in ports. "
                "Use PortsBuilder().with_defaults() or provide a policy_manager port."
            )

        # Create execution context for this DAG run
        pipeline_name = getattr(graph, "name", "unnamed")
        context = ExecutionContext(dag_id=pipeline_name)

        # Fire pipeline started event and check policy
        event = PipelineStarted(
            name=pipeline_name,
            total_waves=len(waves),
            total_nodes=len(graph.nodes),
        )
        await observer_manager.notify(event)

        # Convert to policy context and evaluate policies
        policy_context = self._convert_to_policy_context(event, context)
        policy_response = await self._evaluate_policy_safe(
            policy_manager, policy_context, "pipeline_start"
        )
        if policy_response.signal != PolicySignal.PROCEED:
            reason = policy_response.data if policy_response.data else policy_response.signal.value
            raise OrchestratorError(f"Pipeline start blocked by policy: {reason}")

        for wave_idx, wave in enumerate(waves, 1):
            wave_start_time = time.time()

            # Fire wave started event and check policy
            wave_event = WaveStarted(
                wave_index=wave_idx,
                nodes=wave,
            )
            await observer_manager.notify(wave_event)

            # Convert to policy context and evaluate policies
            wave_policy_context = self._convert_to_policy_context(wave_event, context)
            wave_policy_response = await self._evaluate_policy_safe(
                policy_manager, wave_policy_context, f"wave_{wave_idx}_start"
            )
            if wave_policy_response.signal != PolicySignal.PROCEED:
                reason = (
                    wave_policy_response.data
                    if wave_policy_response.data
                    else wave_policy_response.signal.value
                )
                raise OrchestratorError(f"Wave {wave_idx} execution blocked by policy: {reason}")

            wave_results = await self._execute_wave(
                wave,
                graph,
                node_results,
                initial_input,
                all_ports,
                context=context,
                observer_manager=observer_manager,
                policy_manager=policy_manager,
                wave_index=wave_idx,
                validate=validate,
                **kwargs,
            )
            node_results.update(wave_results)

            # Fire wave completed event (observation only)
            wave_completed = WaveCompleted(
                wave_index=wave_idx,
                duration_ms=(time.time() - wave_start_time) * 1000,
            )
            await observer_manager.notify(wave_completed)

        # Fire pipeline completed event (observation only)
        pipeline_completed = PipelineCompleted(
            name=pipeline_name,
            duration_ms=(time.time() - pipeline_start_time) * 1000,
            node_results=node_results,
        )
        await observer_manager.notify(pipeline_completed)

        return node_results

    async def _execute_wave(
        self,
        wave: list[str],
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        ports: dict[str, Any],
        context: ExecutionContext,
        observer_manager: ObserverManager,
        policy_manager: PolicyManagerPort,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute all nodes in a wave with concurrency limiting."""

        async def execute_with_semaphore(node_name: str) -> tuple[str, Any]:
            async with self._semaphore:
                result = await self._execute_node(
                    node_name,
                    graph,
                    node_results,
                    initial_input,
                    ports,
                    context=context,
                    observer_manager=observer_manager,
                    policy_manager=policy_manager,
                    wave_index=wave_index,
                    validate=validate,
                    **kwargs,
                )
                return node_name, result

        tasks = [execute_with_semaphore(node_name) for node_name in wave]
        wave_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        wave_results = {}
        for output in wave_outputs:
            if isinstance(output, BaseException):
                if isinstance(output, NodeExecutionError):
                    raise output
                raise NodeExecutionError("unknown_node", Exception(str(output))) from output

            node_name, result = output
            wave_results[node_name] = result

        return wave_results

    async def _execute_node(
        self,
        node_name: str,
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        ports: dict[str, Any],
        context: ExecutionContext,
        observer_manager: ObserverManager,
        policy_manager: PolicyManagerPort,
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node.

        The orchestrator provides the MECHANISM for retries through the RETRY signal,
        but the retry POLICY (when to retry, how many times, delays) is determined
        by control handlers registered with the EventBus.
        """
        node_start_time = time.time()
        # Create node context early so it's available in exception handler
        node_context = context.with_node(node_name, wave_index)

        try:
            node_spec = graph.nodes[node_name]
            node_input = self._prepare_node_input(node_spec, node_results, initial_input)

            # Use domain validation
            if validate:
                try:
                    validated_input = node_spec.validate_input(node_input)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    else:
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
            await observer_manager.notify(start_event)

            # Convert to policy context and evaluate policies
            node_policy_context = self._convert_to_policy_context(
                start_event, node_context, node_name
            )
            start_response = await self._evaluate_policy_safe(
                policy_manager, node_policy_context, f"node_{node_name}_start"
            )

            # Handle policy signals
            if start_response.signal == PolicySignal.SKIP:
                # Skip this node and return fallback value if provided
                logger.info(f"Node '{node_name}' skipped by policy")
                return start_response.data
            elif start_response.signal == PolicySignal.FAIL:
                reason = start_response.data if start_response.data else "policy decision"
                raise OrchestratorError(f"Node '{node_name}' execution blocked by policy: {reason}")
            elif start_response.signal != PolicySignal.PROCEED:
                # Treat unexpected signals as errors
                raise OrchestratorError(
                    f"Node '{node_name}' received unexpected policy signal:"
                    f"{start_response.signal.value}"
                )

            # Smart filtering: only pass ports that the function accepts
            sig = inspect.signature(node_spec.fn)
            params = sig.parameters

            # Check if function accepts **kwargs or **ports (catch-all parameters)
            accepts_all_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )

            if accepts_all_kwargs:
                # If function accepts **kwargs/**ports, filter out infrastructure ports
                # unless they're explicitly requested as named parameters
                infrastructure_ports = {"policy_manager", "control_manager"}
                requested_params = set(params.keys())

                # Pass infrastructure ports if explicitly requested, otherwise filter them
                filtered_ports = {}
                for k, v in ports.items():
                    if k in infrastructure_ports and k not in requested_params:
                        continue  # Skip infrastructure ports not explicitly requested
                    filtered_ports[k] = v
            else:
                # Function doesn't accept **kwargs, only pass explicitly named ports
                filtered_ports = {}
                for param_name in params:
                    if param_name in ports and param_name != "input_data":
                        filtered_ports[param_name] = ports[param_name]

            # Execute node function
            raw_output = (
                await node_spec.fn(validated_input, **filtered_ports, **kwargs)
                if asyncio.iscoroutinefunction(node_spec.fn)
                else await asyncio.get_running_loop().run_in_executor(
                    None, lambda: node_spec.fn(validated_input, **filtered_ports, **kwargs)
                )
            )

            # Use domain validation
            if validate:
                try:
                    validated_output = node_spec.validate_output(raw_output)
                except ValidationError as e:
                    if self.strict_validation:
                        raise
                    else:
                        logger.debug("Output validation failed for node '%s': %s", node_name, e)
                        validated_output = raw_output
            else:
                validated_output = raw_output

            # Fire node completed event (observation only)
            complete_event = NodeCompleted(
                name=node_name,
                wave_index=wave_index,
                result=validated_output,
                duration_ms=(time.time() - node_start_time) * 1000,
            )
            await observer_manager.notify(complete_event)

            return validated_output

        except Exception as e:
            # Fire node failed event and check policy
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=e,
            )
            await observer_manager.notify(fail_event)

            # Convert to policy context with error and evaluate policies
            fail_policy_context = self._convert_to_policy_context(
                fail_event, node_context, node_name
            )
            fail_response = await self._evaluate_policy_safe(
                policy_manager, fail_policy_context, f"node_{node_name}_failure"
            )

            # Handle policy signals
            if fail_response.signal == PolicySignal.FALLBACK:
                # Return fallback value instead of failing
                logger.info(
                    f"Node '{node_name}' using fallback value after error: {type(e).__name__}"
                )
                return fail_response.data
            elif fail_response.signal == PolicySignal.RETRY:
                # RETRY signal indicates the policy wants to retry
                # Extract retry metadata safely
                retry_metadata = {}
                if fail_response.metadata and isinstance(fail_response.metadata, dict):
                    retry_metadata = fail_response.metadata

                delay = retry_metadata.get("delay", 0)
                attempt_info = retry_metadata.get("attempt", context.attempt + 1)

                if delay > 0:
                    logger.info(
                        f"Node '{node_name}' retry delayed by {delay}s (attempt {attempt_info})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.info(f"Node '{node_name}' retrying immediately (attempt {attempt_info})")

                # Recursive call to retry the node execution
                # The policy tracks attempts and decides when to stop
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
            else:
                # Default: propagate the error with enhanced message
                logger.error(
                    f"Node '{node_name}' failed with no recovery policy: {type(e).__name__}: {e}"
                )
                raise NodeExecutionError(node_name, e) from e

    def _prepare_node_input(
        self, node_spec: NodeSpec, node_results: dict[str, Any], initial_input: Any
    ) -> Any:
        """Prepare input data for node execution with simplified data mapping.

        Args
        ----
            node_spec: Node specification
            node_results: Results from previously executed nodes
            initial_input: Initial input data for the pipeline

        Returns
        -------
            Prepared input data for the node
        """
        if not node_spec.deps:
            # No dependencies - use initial input
            return initial_input

        if len(node_spec.deps) == 1:
            # Single dependency - pass through directly
            dep_name = next(iter(node_spec.deps))
            return node_results.get(dep_name, initial_input)

        else:
            # Multiple dependencies - preserve namespace structure
            aggregated_data = {}

            # Keep dependency results with their node names as keys
            for dep_name in node_spec.deps:
                if dep_name in node_results:
                    aggregated_data[dep_name] = node_results[dep_name]

            return aggregated_data
