"""DAG Orchestrator - Core execution engine for the Hex-DAG framework.

The Orchestrator walks DirectedGraphs in topological order, executing nodes
concurrently where possible using asyncio.gather().
"""

import asyncio
import time
from typing import Any

from hexai.core.domain.dag import DirectedGraph
from hexai.core.validation import IValidator, ValidationContext, coerce_validator

from ..domain.dag import NodeSpec
from .events import (
    EventBus,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    ObserverManager,
    PipelineCompleted,
    PipelineStarted,
    WaveCompleted,
    WaveStarted,
)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class NodeExecutionError(OrchestratorError):
    """Exception raised when a node fails to execute."""

    def __init__(self, node_name: str, original_error: Exception) -> None:
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")


class ValidationError(OrchestratorError):
    """Exception raised when input validation fails."""

    pass


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
        field_mapping_mode: str = "default",
        validator: "IValidator | None" = None,
    ) -> None:
        """Initialize orchestrator with configuration.

        Args
        ----
            max_concurrent_nodes: Maximum number of nodes to execute concurrently
            ports: Shared ports/dependencies for all pipeline executions
            field_mapping_mode: Schema alignment mode: none/default/vector/custom
            validator: Validator instance to use for input/output validation
        """
        self.max_concurrent_nodes = max_concurrent_nodes
        self._semaphore = asyncio.Semaphore(max_concurrent_nodes)
        self.ports = ports or {}
        self.field_mapping_mode = field_mapping_mode

        # Simplified data mapping - removed complex components
        self._schema_aligner: Any | None = None

        # Initialize validator with default coerce strategy for compatibility
        if validator is None:
            self.validator = coerce_validator()
        else:
            self.validator = validator

    async def run(
        self,
        graph: DirectedGraph,
        initial_input: Any,
        additional_ports: dict[str, Any] | None = None,
        validate: bool = True,
        custom_field_mappings: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a DAG with concurrent processing and resource limits.

        Args
        ----
            custom_field_mappings: Required when field_mapping_mode='custom'
        """
        # Initialize schema aligner based on mode
        if self.field_mapping_mode == "custom":
            if custom_field_mappings is None:
                raise ValueError("custom_field_mappings required when field_mapping_mode='custom'")
            self._schema_aligner = custom_field_mappings  # Simplified
        else:
            # Type-safe cast: field_mapping_mode is FieldMappingMode from __init__
            self._schema_aligner = None  # Simplified

        # Merge orchestrator ports with additional execution-specific ports
        all_ports = {**self.ports}
        if additional_ports:
            all_ports.update(additional_ports)
        if validate:
            try:
                graph.validate()
            except Exception as e:
                raise OrchestratorError(f"Invalid DAG: {e}") from e

        node_results: dict[str, Any] = {}
        waves = graph.waves()
        pipeline_start_time = time.time()

        # Get observers and control bus from ports
        observers: ObserverManager = all_ports.get("observers", ObserverManager())
        control: EventBus = all_ports.get("control_bus", EventBus())

        # Emit pipeline started event
        pipeline_name = getattr(graph, "name", "unnamed")
        event = PipelineStarted(
            name=pipeline_name,
            total_waves=len(waves),
            total_nodes=len(graph.nodes),
        )
        await observers.notify(event)
        if not await control.check(event):
            raise OrchestratorError("Pipeline start vetoed by control handler")

        for wave_idx, wave in enumerate(waves, 1):
            wave_start_time = time.time()

            # Emit wave started event
            wave_event = WaveStarted(
                wave_index=wave_idx,
                nodes=list(wave),  # wave is already a list of node names
            )
            await observers.notify(wave_event)
            if not await control.check(wave_event):
                raise OrchestratorError(f"Wave {wave_idx} vetoed by control handler")

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
            wave_completed = WaveCompleted(
                wave_index=wave_idx,
                duration_ms=(time.time() - wave_start_time) * 1000,
            )
            await observers.notify(wave_completed)

        # Emit pipeline completed event
        pipeline_completed = PipelineCompleted(
            name=pipeline_name,
            duration_ms=(time.time() - pipeline_start_time) * 1000,
            node_results=node_results,
        )
        await observers.notify(pipeline_completed)

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
        # Get observers and control bus from ports
        observers: ObserverManager = ports.get("observers", ObserverManager())
        control: EventBus = ports.get("control_bus", EventBus())

        try:
            node_spec = graph.nodes[node_name]
            node_input = self._prepare_node_input(node_spec, node_results, initial_input)

            # Validate input after preparation if validation is enabled
            # The validation framework will handle type conversion automatically
            if validate and node_spec.in_type is not None:
                validated_input = self._validate_input(node_input, node_spec.in_type, node_name)
            else:
                validated_input = node_input

            # Emit node started event
            start_event = NodeStarted(
                name=node_name,
                wave_index=wave_index,
                dependencies=list(node_spec.deps),
            )
            await observers.notify(start_event)
            if not await control.check(start_event):
                raise OrchestratorError(f"Node '{node_name}' start vetoed by control handler")

            # Execute node function
            raw_output = (
                await node_spec.fn(validated_input, **ports, **kwargs)
                if asyncio.iscoroutinefunction(node_spec.fn)
                else await asyncio.get_event_loop().run_in_executor(
                    None, lambda: node_spec.fn(validated_input, **ports, **kwargs)
                )
            )

            # Validate output if validation is enabled and node has output type specified
            if validate and node_spec.out_type is not None:
                validated_output = self._validate_output(raw_output, node_spec.out_type, node_name)
            else:
                validated_output = raw_output

            # Emit node completed event
            complete_event = NodeCompleted(
                name=node_name,
                wave_index=wave_index,
                result=validated_output,
                duration_ms=(time.time() - node_start_time) * 1000,
            )
            await observers.notify(complete_event)

            return validated_output

        except Exception as e:
            # Emit node failed event
            fail_event = NodeFailed(
                name=node_name,
                wave_index=wave_index,
                error=e,
            )
            await observers.notify(fail_event)
            raise NodeExecutionError(node_name, e) from e

    def _validate_input(self, input_data: Any, expected_type: type, node_name: str) -> Any:
        """Validate input data against expected type using validation framework."""
        # Create validation context with node information
        context = ValidationContext(node_name=node_name, validation_stage="input")

        # Use the validation framework
        result = self.validator.validate_input(input_data, expected_type, context)

        if result.is_valid:
            return result.data
        else:
            # Convert validation errors to orchestrator ValidationError
            error_messages = "; ".join(result.errors)
            raise ValidationError(
                f"Input validation failed for node '{node_name}': {error_messages}"
            )

    def _validate_output(self, output_data: Any, expected_type: type, node_name: str) -> Any:
        """Validate output data against expected type using validation framework."""
        # Create validation context with node information
        context = ValidationContext(node_name=node_name, validation_stage="output")

        # Use the validation framework
        result = self.validator.validate_output(output_data, expected_type, context)

        if result.is_valid:
            return result.data
        else:
            # Convert validation errors to orchestrator ValidationError
            error_messages = "; ".join(result.errors)
            raise ValidationError(
                f"Output validation failed for node '{node_name}': {error_messages}"
            )

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

        # Check for explicit input mapping in node parameters
        input_mapping = node_spec.params.get("input_mapping")
        if input_mapping:
            # Use explicit data mapping - simple dictionary mapping
            mapped_data = {}

            # Apply input mapping rules
            for output_key, input_sources in input_mapping.items():
                if isinstance(input_sources, list):
                    # Multiple sources - use first available
                    for source in input_sources:
                        if isinstance(source, str) and source in node_results:
                            mapped_data[output_key] = node_results[source]
                            break
                        elif isinstance(source, dict) and "node" in source and "field" in source:
                            node_name = source["node"]
                            field_name = source["field"]
                            if node_name in node_results and field_name in node_results[node_name]:
                                mapped_data[output_key] = node_results[node_name][field_name]
                                break
                elif isinstance(input_sources, str):
                    # Single source - handle dotted notation like "processor.text"
                    if "." in input_sources:
                        # Handle dotted notation for field access
                        parts = input_sources.split(".")
                        node_name = parts[0]
                        field_path = parts[1:]

                        if node_name in node_results:
                            current_value = node_results[node_name]
                            # Navigate nested field path
                            for field in field_path:
                                if hasattr(current_value, field):
                                    current_value = getattr(current_value, field)
                                elif isinstance(current_value, dict) and field in current_value:
                                    current_value = current_value[field]
                                else:
                                    current_value = None
                                    break
                            if current_value is not None:
                                mapped_data[output_key] = current_value
                    elif input_sources in node_results:
                        # Direct node name
                        mapped_data[output_key] = node_results[input_sources]
                elif (
                    isinstance(input_sources, dict)
                    and "node" in input_sources
                    and "field" in input_sources
                ):
                    # Specific field from specific node
                    node_name = input_sources["node"]
                    field_name = input_sources["field"]
                    if node_name in node_results and field_name in node_results[node_name]:
                        mapped_data[output_key] = node_results[node_name][field_name]

            # Include any unmapped initial input fields
            if isinstance(initial_input, dict):
                for key, value in initial_input.items():
                    if key not in mapped_data:
                        mapped_data[key] = value

            return mapped_data

        elif len(node_spec.deps) == 1:
            # Single dependency - pass through directly
            dep_name = next(iter(node_spec.deps))
            return node_results.get(dep_name, initial_input)

        else:
            # Multiple dependencies - preserve namespace structure
            aggregated_data = {}

            # Keep dependency results with their node names as keys
            for dep_name in node_spec.deps:
                if dep_name in node_results:
                    value = node_results[dep_name]
                    # If the node expects dict input and we have Pydantic models, convert them
                    if node_spec.in_type is dict and hasattr(value, "model_dump"):
                        aggregated_data[dep_name] = value.model_dump()
                    else:
                        aggregated_data[dep_name] = value

            return aggregated_data
