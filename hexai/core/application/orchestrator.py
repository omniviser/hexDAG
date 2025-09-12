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
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    PipelineCompletedEvent,
    PipelineStartedEvent,
    WaveCompletedEvent,
    WaveStartedEvent,
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

        event_manager = all_ports.get("event_manager")

        # Emit pipeline started event using elegant null-safe syntax
        if event_manager:
            await event_manager.emit(
                PipelineStartedEvent(
                    pipeline_name=getattr(graph, "name", "unnamed"),
                    total_waves=len(waves),
                    total_nodes=len(graph.nodes),
                )
            )

        for wave_idx, wave in enumerate(waves, 1):
            wave_start_time = time.time()

            # Emit wave started event
            if event_manager:
                await event_manager.emit(WaveStartedEvent(wave_index=wave_idx, nodes=wave))

            wave_results = await self._execute_wave(
                wave,
                graph,
                node_results,
                initial_input,
                all_ports,
                wave_index=wave_idx,
                validate=validate,
                **kwargs,
            )
            node_results.update(wave_results)

            # Emit wave completed event
            if event_manager:
                await event_manager.emit(
                    WaveCompletedEvent(
                        wave_index=wave_idx,
                        nodes=wave,
                        execution_time=time.time() - wave_start_time,
                    )
                )

        # Emit pipeline completed event
        if event_manager:
            await event_manager.emit(
                PipelineCompletedEvent(
                    pipeline_name=getattr(graph, "name", "unnamed"),
                    total_execution_time=time.time() - pipeline_start_time,
                    node_results=node_results,
                )
            )

        return node_results

    async def _execute_wave(
        self,
        wave: list[str],
        graph: DirectedGraph,
        node_results: dict[str, Any],
        initial_input: Any,
        ports: dict[str, Any],
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
        wave_index: int = 0,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a single node."""
        node_start_time = time.time()
        event_manager = ports.get("event_manager")

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
                        logger.debug(f"Input validation failed for node '{node_name}': {e}")
                        validated_input = node_input
            else:
                validated_input = node_input

            # Emit node started event
            if event_manager:
                await event_manager.emit(
                    NodeStartedEvent(
                        node_name=node_name,
                        wave_index=wave_index,
                        dependencies=list(node_spec.deps),
                    )
                )

            # Execute node function
            raw_output = (
                await node_spec.fn(validated_input, **ports, **kwargs)
                if asyncio.iscoroutinefunction(node_spec.fn)
                else await asyncio.get_event_loop().run_in_executor(
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
                        logger.debug(f"Output validation failed for node '{node_name}': {e}")
                        validated_output = raw_output
            else:
                validated_output = raw_output

            # Emit node completed event
            if event_manager:
                await event_manager.emit(
                    NodeCompletedEvent(
                        node_name=node_name,
                        result=validated_output,
                        execution_time=time.time() - node_start_time,
                        wave_index=wave_index,
                    )
                )

            return validated_output

        except Exception as e:
            # Emit node failed event
            if event_manager:
                await event_manager.emit(
                    NodeFailedEvent(node_name=node_name, error=e, wave_index=wave_index)
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
