"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import logging
import time
from typing import Any

from hexai.core.domain.dag import DirectedGraph, NodeSpec, ValidationError

from .events import (
    ControlManager,
    ControlSignal,
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
        ports: dict[str, Any] | None = None,
        strict_validation: bool = False,
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Shared ports/dependencies for all pipeline executions
            strict_validation: If True, raise errors on validation failure
        """
        self.max_concurrent_nodes = max_concurrent_nodes
        self._semaphore = asyncio.Semaphore(max_concurrent_nodes)
        self.ports = ports or {}
        self.strict_validation = strict_validation

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
        all_ports = {**self.ports}
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

        # Get observer manager and control manager from ports
        observer_manager: ObserverManager = all_ports.get("observer_manager", ObserverManager())
        control_manager: ControlManager = all_ports.get("control_manager", ControlManager())

        # Create execution context for this DAG run
        pipeline_name = getattr(graph, "name", "unnamed")
        context = ExecutionContext(dag_id=pipeline_name)

        # Fire pipeline started event and check control
        event = PipelineStarted(
            name=pipeline_name,
            total_waves=len(waves),
            total_nodes=len(graph.nodes),
        )
        await observer_manager.notify(event)
        control_response = await control_manager.check(event, context)
        if control_response.signal != ControlSignal.PROCEED:
            raise OrchestratorError(f"Pipeline start blocked: {control_response.signal.value}")

        for wave_idx, wave in enumerate(waves, 1):
            wave_start_time = time.time()

            # Fire wave started event and check control
            wave_event = WaveStarted(
                wave_index=wave_idx,
                nodes=wave,
            )
            await observer_manager.notify(wave_event)
            wave_response = await control_manager.check(wave_event, context)
            if wave_response.signal != ControlSignal.PROCEED:
                raise OrchestratorError(f"Wave {wave_idx} blocked: {wave_response.signal.value}")

            wave_results = await self._execute_wave(
                wave,
                graph,
                node_results,
                initial_input,
                all_ports,
                context=context,
                observer_manager=observer_manager,
                control_manager=control_manager,
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
        control_manager: ControlManager,
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
                    control_manager=control_manager,
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
        control_manager: ControlManager,
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

            # Fire node started event and check control
            start_event = NodeStarted(
                name=node_name,
                wave_index=wave_index,
                dependencies=list(node_spec.deps),
            )
            await observer_manager.notify(start_event)
            start_response = await control_manager.check(start_event, node_context)

            # Handle control signals
            if start_response.signal == ControlSignal.SKIP:
                # Skip this node and return fallback value if provided
                return start_response.data
            elif start_response.signal == ControlSignal.FAIL:
                raise OrchestratorError(f"Node '{node_name}' blocked: {start_response.data}")
            elif start_response.signal != ControlSignal.PROCEED:
                # For now, treat other signals as errors
                raise OrchestratorError(
                    f"Node '{node_name}' blocked: {start_response.signal.value}"
                )

            # Execute node function
            raw_output = (
                await node_spec.fn(validated_input, **ports, **kwargs)
                if asyncio.iscoroutinefunction(node_spec.fn)
                else await asyncio.get_running_loop().run_in_executor(
                    None, lambda: node_spec.fn(validated_input, **ports, **kwargs)
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
            # Fire node failed event and check control
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=e,
            )
            await observer_manager.notify(fail_event)
            fail_response = await control_manager.check(fail_event, node_context)

            # Handle control signals
            if fail_response.signal == ControlSignal.FALLBACK:
                # Return fallback value instead of failing
                return fail_response.data
            elif fail_response.signal == ControlSignal.RETRY:
                # RETRY signal indicates the policy wants to retry
                # The orchestrator enables this by re-executing the node
                # The retry policy (attempts, delays) is managed by the handler
                retry_data = fail_response.data if isinstance(fail_response.data, dict) else {}
                delay = retry_data.get("delay", 0)

                if delay > 0:
                    await asyncio.sleep(delay)

                # Recursive call to retry the node execution
                # The policy handler tracks attempts and decides when to stop
                return await self._execute_node(
                    node_name=node_name,
                    graph=graph,
                    node_results=node_results,
                    initial_input=initial_input,
                    ports=ports,
                    context=context.with_attempt(context.attempt + 1),
                    observer_manager=observer_manager,
                    control_manager=control_manager,
                    wave_index=wave_index,
                    validate=validate,
                    **kwargs,
                )
            else:
                # Default: propagate the error
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
