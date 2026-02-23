"""Pipeline execution API.

Provides unified functions for executing hexDAG pipelines.
Delegates to :class:`~hexdag.kernel.pipeline_runner.PipelineRunner` for the
heavy lifting (YAML parsing, port instantiation, secret loading, execution).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def execute(
    yaml_content: str,
    inputs: dict[str, Any] | None = None,
    ports: dict[str, Any] | None = None,
    node_ports: dict[str, dict[str, Any]] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute a YAML pipeline.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    inputs : dict | None
        Input values for the pipeline
    ports : dict | None
        Global port adapters (llm, memory, etc.). If None, uses mock adapters.
    node_ports : dict | None
        Per-node port overrides. Dict of node_name -> {port_type: adapter_instance}.
        Example: {"analyzer_a": {"llm": MockLLM(responses="custom")}}
    timeout : float
        Execution timeout in seconds (default: 30.0)

    Returns
    -------
    dict
        Execution result with keys:
        - success: bool - Whether execution succeeded
        - nodes: list - Node execution results
        - final_output: Any - Final pipeline output
        - error: str | None - Error message if failed
        - duration_ms: float - Execution time in milliseconds

    Examples
    --------
    >>> import asyncio
    >>> result = asyncio.run(execute('''
    ... apiVersion: hexdag/v1
    ... kind: Pipeline
    ... metadata:
    ...   name: test
    ... spec:
    ...   nodes:
    ...     - kind: data_node
    ...       metadata:
    ...         name: start
    ...       spec:
    ...         output:
    ...           value: "hello"
    ...       dependencies: []
    ... '''))
    >>> result["success"]
    True
    """
    if inputs is None:
        inputs = {}

    start = time.perf_counter()
    node_results: list[dict[str, Any]] = []

    try:
        # Use PipelineRunner for non-streaming execution when no per-node ports
        if node_ports is None:
            from hexdag.kernel.pipeline_runner import PipelineRunner

            # If explicit ports provided, use them as overrides;
            # otherwise let PipelineRunner create mocks / use YAML config
            port_overrides = ports if ports is not None else _create_default_ports()

            runner = PipelineRunner(port_overrides=port_overrides)

            # Build for wave info (needed for response format)
            graph, _config = runner._build(yaml_content=yaml_content)
            waves = graph.waves()
            execution_order = [node for wave in waves for node in wave]

            result = await asyncio.wait_for(
                runner.run_from_string(yaml_content, input_data=inputs),
                timeout=timeout,
            )
            success = True
            final_output = result
            error = None

            node_results.extend(
                {
                    "name": node_name,
                    "status": "completed",
                    "output": result.get(node_name) if isinstance(result, dict) else None,
                    "duration_ms": None,
                }
                for node_name in execution_order
            )
        else:
            # Per-node ports require direct Orchestrator usage (PortsConfiguration)
            return await _execute_with_node_ports(yaml_content, inputs, ports, node_ports, timeout)

    except TimeoutError:
        success = False
        final_output = None
        error = f"Execution timed out after {timeout}s"
    except Exception as e:
        success = False
        final_output = None
        error = f"Pipeline failed: {e}"

    duration = (time.perf_counter() - start) * 1000

    return {
        "success": success,
        "nodes": node_results,
        "final_output": final_output,
        "error": error,
        "duration_ms": duration,
    }


def dry_run(yaml_content: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze pipeline without executing.

    Returns execution plan, dependency order, and wave structure.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    inputs : dict | None
        Input values (used for analysis, not execution)

    Returns
    -------
    dict
        Analysis result with keys:
        - valid: bool - Whether pipeline is valid
        - execution_order: list[str] - Node execution order
        - node_count: int - Number of nodes
        - waves: list[list[str]] - Execution waves (parallel groups)
        - dependency_map: dict - Node dependencies and wave assignments
        - error: str | None - Error message if invalid

    Examples
    --------
    >>> result = dry_run('''
    ... apiVersion: hexdag/v1
    ... kind: Pipeline
    ... metadata:
    ...   name: test
    ... spec:
    ...   nodes:
    ...     - kind: data_node
    ...       metadata:
    ...         name: a
    ...       spec:
    ...         output:
    ...           value: 1
    ...       dependencies: []
    ...     - kind: data_node
    ...       metadata:
    ...         name: b
    ...       spec:
    ...         output:
    ...           value: 2
    ...       dependencies: [a]
    ... ''')
    >>> result["valid"]
    True
    >>> result["node_count"]
    2
    """
    try:
        from hexdag.compiler import YamlPipelineBuilder

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Get execution order from waves
        waves = graph.waves()
        execution_order = [node for wave in waves for node in wave]

        # Analyze dependencies
        dependency_map: dict[str, dict[str, Any]] = {}
        for node_id in execution_order:
            node = graph.nodes.get(node_id)
            dependency_map[node_id] = {
                "dependencies": list(node.deps) if node else [],
                "wave": next((i for i, wave in enumerate(waves) if node_id in wave), -1),
            }

        return {
            "valid": True,
            "execution_order": execution_order,
            "node_count": len(execution_order),
            "waves": waves,
            "dependency_map": dependency_map,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


def create_ports_from_config(port_config: dict[str, Any]) -> dict[str, Any]:
    """Create adapter instances from port configuration.

    Accepts adapter aliases (``MockLLM``, ``mock_llm``, ``llm:mock``)
    and full module paths. Alias resolution is handled by the resolver
    via ``ComponentInstantiator``. Always ensures ``llm`` and ``memory``
    defaults are present.

    Parameters
    ----------
    port_config : dict
        Dict of port_name -> {adapter: str, config: dict}

    Returns
    -------
    dict
        Dict of port_name -> adapter_instance

    Examples
    --------
    >>> ports = create_ports_from_config({
    ...     "llm": {"adapter": "MockLLM", "config": {"responses": "hello"}},
    ...     "memory": {"adapter": "InMemoryMemory", "config": {}}
    ... })
    >>> "llm" in ports
    True
    """
    from hexdag.compiler.component_instantiator import ComponentInstantiator

    instantiator = ComponentInstantiator()
    try:
        ports = instantiator.instantiate_ports(port_config)
    except Exception:
        # Fallback to defaults on failure (backward-compatible behavior)
        ports = _create_default_ports()

    # Ensure we always have llm and memory defaults
    if "llm" not in ports:
        from hexdag.stdlib.adapters.mock import MockLLM

        ports["llm"] = MockLLM()
    if "memory" not in ports:
        from hexdag.stdlib.adapters.memory import InMemoryMemory

        ports["memory"] = InMemoryMemory()

    return ports


def _create_default_ports() -> dict[str, Any]:
    """Create default mock ports for execution."""
    from hexdag.kernel.ports.tool_router import ToolRouter
    from hexdag.stdlib.adapters.memory import InMemoryMemory
    from hexdag.stdlib.adapters.mock import MockLLM

    return {
        "llm": MockLLM(),
        "memory": InMemoryMemory(),
        "tool_router": ToolRouter(),
    }


async def _execute_with_node_ports(
    yaml_content: str,
    inputs: dict[str, Any],
    ports: dict[str, Any] | None,
    node_ports: dict[str, dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    """Execute with per-node port overrides (requires PortsConfiguration)."""
    start = time.perf_counter()
    node_results: list[dict[str, Any]] = []

    try:
        from hexdag.compiler import YamlPipelineBuilder
        from hexdag.kernel.orchestration.models import PortConfig, PortsConfiguration
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        if ports is None:
            ports = _create_default_ports()

        global_port_configs = {
            port_name: PortConfig(port=port_instance) for port_name, port_instance in ports.items()
        }
        node_port_configs = {
            node_name: {
                port_type: PortConfig(port=port_instance)
                for port_type, port_instance in node_port_dict.items()
            }
            for node_name, node_port_dict in node_ports.items()
        }

        ports_config = PortsConfiguration(
            global_ports=global_port_configs,
            node_ports=node_port_configs,
        )
        orchestrator = Orchestrator(ports=ports_config)

        waves = graph.waves()
        execution_order = [node for wave in waves for node in wave]

        try:
            result = await asyncio.wait_for(
                orchestrator.run(graph, inputs),
                timeout=timeout,
            )
            success = True
            final_output = result
            error = None

            node_results.extend(
                {
                    "name": node_name,
                    "status": "completed",
                    "output": result.get(node_name) if isinstance(result, dict) else None,
                    "duration_ms": None,
                }
                for node_name in execution_order
            )
        except TimeoutError:
            success = False
            final_output = None
            error = f"Execution timed out after {timeout}s"

    except Exception as e:
        success = False
        final_output = None
        error = f"Pipeline failed: {e}"

    duration = (time.perf_counter() - start) * 1000

    return {
        "success": success,
        "nodes": node_results,
        "final_output": final_output,
        "error": error,
        "duration_ms": duration,
    }


async def execute_streaming(
    yaml_content: str,
    inputs: dict[str, Any] | None = None,
    ports: dict[str, Any] | None = None,
    node_ports: dict[str, dict[str, Any]] | None = None,
    timeout: float = 30.0,
) -> AsyncGenerator[dict[str, Any]]:
    """Execute a YAML pipeline with real-time streaming of events.

    This function yields execution events as they occur, allowing for
    real-time UI updates during pipeline execution.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    inputs : dict | None
        Input values for the pipeline
    ports : dict | None
        Global port adapters (llm, memory, etc.). If None, uses mock adapters.
    node_ports : dict | None
        Per-node port overrides. Dict of node_name -> {port_type: adapter_instance}.
    timeout : float
        Execution timeout in seconds (default: 30.0)

    Yields
    ------
    dict
        Event dictionaries with keys:
        - event: str - Event type (plan, wave_start, node_start, node_complete,
          node_failed, complete, error)
        - data: dict - Event-specific data

    Examples
    --------
    >>> async for event in execute_streaming(yaml_content):  # doctest: +SKIP
    ...     if event["event"] == "node_start":
    ...         print(f"Starting: {event['data']['name']}")
    ...     elif event["event"] == "node_complete":
    ...         print(f"Completed: {event['data']['name']}")
    """
    if inputs is None:
        inputs = {}

    start_time = time.perf_counter()

    try:
        from hexdag.compiler import YamlPipelineBuilder
        from hexdag.drivers.observer_manager import LocalObserverManager
        from hexdag.kernel.orchestration.events import (
            NodeCompleted,
            NodeFailed,
            NodeStarted,
            PipelineCompleted,
            WaveCompleted,
            WaveStarted,
        )
        from hexdag.kernel.orchestration.models import PortConfig, PortsConfiguration
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        waves = graph.waves()
        total_nodes = sum(len(wave) for wave in waves)
        node_names = [node for wave in waves for node in wave]

        yield {
            "event": "plan",
            "data": {
                "waves": waves,
                "total_nodes": total_nodes,
                "node_names": node_names,
            },
        }

        if ports is None:
            ports = _create_default_ports()

        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def queue_observer(event: Any) -> None:
            """Observer that pushes events to the async queue."""
            if isinstance(event, WaveStarted):
                await event_queue.put({
                    "event": "wave_start",
                    "data": {
                        "wave_index": event.wave_index,
                        "nodes": event.nodes,
                    },
                })
            elif isinstance(event, NodeStarted):
                await event_queue.put({
                    "event": "node_start",
                    "data": {
                        "name": event.name,
                        "wave_index": event.wave_index,
                    },
                })
            elif isinstance(event, NodeCompleted):
                await event_queue.put({
                    "event": "node_complete",
                    "data": {
                        "name": event.name,
                        "wave_index": event.wave_index,
                        "output": event.result,
                        "duration_ms": event.duration_ms,
                    },
                })
            elif isinstance(event, NodeFailed):
                await event_queue.put({
                    "event": "node_failed",
                    "data": {
                        "name": event.name,
                        "wave_index": event.wave_index,
                        "error": str(event.error),
                    },
                })
            elif isinstance(event, WaveCompleted):
                await event_queue.put({
                    "event": "wave_complete",
                    "data": {
                        "wave_index": event.wave_index,
                        "duration_ms": event.duration_ms,
                    },
                })
            elif isinstance(event, PipelineCompleted):
                await event_queue.put(None)

        observer_manager = LocalObserverManager()
        observer_manager.register(
            queue_observer,
            observer_id="streaming_observer",
            event_types=[
                WaveStarted,
                NodeStarted,
                NodeCompleted,
                NodeFailed,
                WaveCompleted,
                PipelineCompleted,
            ],
        )

        if node_ports:
            global_port_configs = {
                port_name: PortConfig(port=port_instance)
                for port_name, port_instance in ports.items()
            }
            global_port_configs["observer_manager"] = PortConfig(port=observer_manager)

            node_port_configs = {
                node_name: {
                    port_type: PortConfig(port=port_instance)
                    for port_type, port_instance in node_port_dict.items()
                }
                for node_name, node_port_dict in node_ports.items()
            }

            ports_config = PortsConfiguration(
                global_ports=global_port_configs,
                node_ports=node_port_configs,
            )
            orchestrator = Orchestrator(ports=ports_config)
        else:
            ports["observer_manager"] = observer_manager
            orchestrator = Orchestrator(ports=ports)

        node_results: dict[str, dict[str, Any]] = {}
        execution_error: str | None = None
        final_output: Any = None
        success = True

        async def run_pipeline() -> None:
            """Run the pipeline in background and signal completion."""
            nonlocal final_output, execution_error, success
            try:
                result = await asyncio.wait_for(
                    orchestrator.run(graph, inputs),
                    timeout=timeout,
                )
                final_output = result
            except TimeoutError:
                execution_error = f"Execution timed out after {timeout}s"
                success = False
            except Exception as e:
                execution_error = str(e)
                success = False
            finally:
                await event_queue.put(None)

        pipeline_task = asyncio.create_task(run_pipeline())

        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break

                if event["event"] == "node_complete":
                    node_results[event["data"]["name"]] = {
                        "status": "completed",
                        "output": event["data"].get("output"),
                        "duration_ms": event["data"].get("duration_ms"),
                    }
                elif event["event"] == "node_failed":
                    node_results[event["data"]["name"]] = {
                        "status": "failed",
                        "error": event["data"].get("error"),
                    }
                    success = False

                yield event
        finally:
            await pipeline_task

        duration_ms = (time.perf_counter() - start_time) * 1000
        yield {
            "event": "complete",
            "data": {
                "success": success,
                "final_output": final_output,
                "error": execution_error,
                "duration_ms": duration_ms,
                "nodes": [{"name": name, **data} for name, data in node_results.items()],
            },
        }

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        yield {
            "event": "error",
            "data": {
                "error": str(e),
                "duration_ms": duration_ms,
            },
        }
