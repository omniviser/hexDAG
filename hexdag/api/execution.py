"""Pipeline execution API.

Provides unified functions for executing hexDAG pipelines.
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
        from hexdag.core.orchestration.models import PortConfig, PortsConfiguration
        from hexdag.core.orchestration.orchestrator import Orchestrator
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        # Build pipeline
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Create ports (use provided or default to mocks)
        if ports is None:
            ports = _create_default_ports()

        # Create PortsConfiguration if we have per-node ports
        if node_ports:
            # Convert flat ports dict to PortConfig objects for global ports
            global_port_configs = {
                port_name: PortConfig(port=port_instance)
                for port_name, port_instance in ports.items()
            }

            # Convert node_ports to PortConfig objects
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
            orchestrator = Orchestrator(ports=ports)

        # Get execution order for reporting
        waves = graph.waves()
        execution_order = [node for wave in waves for node in wave]

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                orchestrator.run(graph, inputs),
                timeout=timeout,
            )
            success = True
            final_output = result
            error = None

            # Build node results from execution order
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
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

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


def _create_default_ports() -> dict[str, Any]:
    """Create default mock ports for execution."""
    from hexdag.builtin.adapters.memory import InMemoryMemory
    from hexdag.builtin.adapters.mock import MockLLM

    return {
        "llm": MockLLM(),
        "memory": InMemoryMemory(),
    }


def create_ports_from_config(port_config: dict[str, Any]) -> dict[str, Any]:
    """Create adapter instances from port configuration.

    Uses a registry-based approach for adapter lookup.

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
    from hexdag.builtin.adapters.memory import InMemoryMemory
    from hexdag.builtin.adapters.mock import MockLLM

    ports: dict[str, Any] = {}

    for port_name, config in port_config.items():
        adapter_name = config.get("adapter", "")
        adapter_config = config.get("config", {})
        adapter = _create_adapter(adapter_name, adapter_config, port_name)
        if adapter:
            ports[port_name] = adapter

    # Ensure we always have llm and memory ports
    if "llm" not in ports:
        ports["llm"] = MockLLM()
    if "memory" not in ports:
        ports["memory"] = InMemoryMemory()

    return ports


# Cache for discovered adapters (populated lazily)
_adapter_cache: dict[str, str] | None = None


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case, handling acronyms like OpenAI, LLM.

    Examples:
        AzureOpenAIAdapter -> azure_openai_adapter
        MockLLM -> mock_llm
        InMemoryMemory -> in_memory_memory
    """
    import re

    # First, handle known acronyms by preserving them as single units
    # e.g., OpenAI -> Openai, LLM -> Llm (temporarily)
    name = re.sub(r"OpenAI", "Openai", name)
    name = re.sub(r"LLM", "Llm", name)
    name = re.sub(r"API", "Api", name)
    name = re.sub(r"DB", "Db", name)
    name = re.sub(r"SQL", "Sql", name)

    # Now do standard CamelCase to snake_case conversion
    # Insert underscore before uppercase letters (except at start)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _get_adapter_registry() -> dict[str, str]:
    """Get adapter registry mapping normalized names to module paths.

    Dynamically discovers adapters from:
    1. hexdag.builtin.adapters.* (builtin adapters)
    2. hexdag_plugins.* (plugin adapters)

    Returns a dict of normalized_name -> module_path.class_name
    """
    global _adapter_cache
    if _adapter_cache is not None:
        return _adapter_cache

    _adapter_cache = {}

    # Discover builtin adapters via hexdag.api.components
    try:
        from hexdag.api.components import list_adapters

        for adapter_info in list_adapters():
            name = adapter_info["name"]
            module_path = adapter_info["module_path"]

            # Add multiple lookup keys for convenience
            _adapter_cache[name.lower()] = module_path
            # Also add without "Adapter" suffix
            if name.endswith("Adapter"):
                short_name = name[:-7].lower()
                _adapter_cache[short_name] = module_path
                # Also add snake_case of short name (e.g., azure_openai)
                snake_short = _to_snake_case(name[:-7])
                _adapter_cache[snake_short] = module_path
            # Also add snake_case version of full name
            snake_name = _to_snake_case(name)
            _adapter_cache[snake_name] = module_path

    except ImportError:
        pass

    # Discover plugin adapters from hexdag_plugins
    _discover_plugin_adapters(_adapter_cache)

    return _adapter_cache


def _discover_plugin_adapters(cache: dict[str, str]) -> None:
    """Discover adapters from hexdag_plugins namespace.

    Plugin Convention
    -----------------
    Plugins in hexdag_plugins follow this structure::

        hexdag_plugins/
        ├── <plugin_name>/
        │   ├── __init__.py          # Re-exports from adapters/nodes/tools
        │   ├── adapters/            # Adapter implementations
        │   │   ├── __init__.py
        │   │   └── my_adapter.py
        │   ├── nodes/               # Custom node types
        │   │   ├── __init__.py
        │   │   └── my_node.py
        │   ├── tools/               # Agent tools
        │   │   ├── __init__.py
        │   │   └── my_tool.py
        │   └── ports/               # Custom port protocols (optional)
        │       └── __init__.py

    Adapters MUST:
    1. Inherit from their port protocol (LLM, Memory, SecretPort, etc.)
    2. Be exported from plugin's __init__.py or adapters/__init__.py
    """
    import importlib
    import pkgutil

    def _scan_module_for_adapters(module_path: str) -> None:
        """Scan a module for adapter classes and add to cache."""
        try:
            plugin_module = importlib.import_module(module_path)
        except ImportError:
            return

        for attr_name in dir(plugin_module):
            if attr_name.startswith("_"):
                continue

            obj = getattr(plugin_module, attr_name, None)
            if obj is None or not isinstance(obj, type):
                continue

            # Check if it looks like an adapter (ends with Adapter, Memory, or LLM)
            if not (
                attr_name.endswith("Adapter")
                or attr_name.endswith("Memory")
                or attr_name.endswith("LLM")
            ):
                continue

            full_path = f"{module_path}.{attr_name}"

            # Add multiple lookup keys
            cache[attr_name.lower()] = full_path

            # Handle different naming patterns
            if attr_name.endswith("Adapter"):
                short_name = attr_name[:-7].lower()
                cache[short_name] = full_path
                snake_short = _to_snake_case(attr_name[:-7])
                cache[snake_short] = full_path
            elif attr_name.endswith("Memory"):
                short_name = attr_name[:-6].lower()
                cache[short_name] = full_path
            elif attr_name.endswith("LLM"):
                short_name = attr_name[:-3].lower()
                cache[short_name] = full_path

            # Snake_case version of full name
            snake_name = _to_snake_case(attr_name)
            cache[snake_name] = full_path

    try:
        import hexdag_plugins

        for module_info in pkgutil.iter_modules(hexdag_plugins.__path__):
            plugin_name = module_info.name

            # Scan structured layout: hexdag_plugins/<plugin>/adapters/
            _scan_module_for_adapters(f"hexdag_plugins.{plugin_name}.adapters")

            # Scan flat layout: hexdag_plugins/<plugin>/
            _scan_module_for_adapters(f"hexdag_plugins.{plugin_name}")

    except ImportError:
        # hexdag_plugins not installed
        pass


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${VAR_NAME} references in config values from environment.

    Supports:
    - ${VAR_NAME} - resolves to environment variable or empty string
    - Nested dicts are resolved recursively
    """
    import os

    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            resolved[key] = os.environ.get(var_name, "")
        elif isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        else:
            resolved[key] = value
    return resolved


def _coerce_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """Coerce string values in config to appropriate types.

    YAML config values may come as strings (e.g., delay_seconds: "1")
    but adapters expect proper types. This function converts:
    - Numeric strings to int/float
    - "true"/"false" strings to bool
    """
    coerced: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str):
            # Try to convert to number
            try:
                if "." in value:
                    coerced[key] = float(value)
                else:
                    coerced[key] = int(value)
                continue
            except ValueError:
                pass

            # Try to convert to bool
            if value.lower() == "true":
                coerced[key] = True
                continue
            if value.lower() == "false":
                coerced[key] = False
                continue

            # Keep as string
            coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def _create_adapter(adapter_name: str, adapter_config: dict[str, Any], port_name: str) -> Any:
    """Create an adapter instance from name and config.

    Uses dynamic adapter discovery from:
    1. hexdag.builtin.adapters.* (builtin adapters)
    2. hexdag_plugins.* (plugin adapters)
    """
    import importlib

    from hexdag.builtin.adapters.memory import InMemoryMemory
    from hexdag.builtin.adapters.mock import MockLLM

    # Resolve environment variables first (e.g., ${OPENAI_API_KEY})
    adapter_config = _resolve_env_vars(adapter_config)
    # Then coerce string values to appropriate types
    adapter_config = _coerce_config_values(adapter_config)

    adapter_lower = adapter_name.lower()

    # Get dynamic adapter registry
    registry = _get_adapter_registry()

    # Look up in registry
    if adapter_lower in registry:
        full_path = registry[adapter_lower]
        try:
            # Parse module path and class name
            module_path, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            return adapter_cls(**adapter_config)
        except (ImportError, AttributeError, ValueError) as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to instantiate adapter {adapter_name}: {e}"
            )
            # Fallback to mock for LLM adapters
            if (
                "llm" in port_name.lower()
                or "openai" in adapter_lower
                or "anthropic" in adapter_lower
            ):
                return MockLLM(**adapter_config)
            return None

    # Fallback: guess from port name
    if "llm" in port_name.lower():
        return MockLLM(**adapter_config)
    if "memory" in port_name.lower():
        return InMemoryMemory(**adapter_config)

    return None


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
        from hexdag.builtin.adapters.local import LocalObserverManager
        from hexdag.core.orchestration.events import (
            NodeCompleted,
            NodeFailed,
            NodeStarted,
            PipelineCompleted,
            WaveCompleted,
            WaveStarted,
        )
        from hexdag.core.orchestration.models import PortConfig, PortsConfiguration
        from hexdag.core.orchestration.orchestrator import Orchestrator
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        # Build pipeline
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Get waves for plan event
        waves = graph.waves()
        total_nodes = sum(len(wave) for wave in waves)
        node_names = [node for wave in waves for node in wave]

        # Yield plan event
        yield {
            "event": "plan",
            "data": {
                "waves": waves,
                "total_nodes": total_nodes,
                "node_names": node_names,
            },
        }

        # Create ports (use provided or default to mocks)
        if ports is None:
            ports = _create_default_ports()

        # Create observer manager with async queue for event streaming
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
                # Signal completion
                await event_queue.put(None)

        # Create observer manager and register our queue observer
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

        # Create PortsConfiguration if we have per-node ports
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
            # Add observer_manager to ports
            ports["observer_manager"] = observer_manager
            orchestrator = Orchestrator(ports=ports)

        # Track node results for final summary
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
                # Signal end of events
                await event_queue.put(None)

        # Start pipeline execution in background
        pipeline_task = asyncio.create_task(run_pipeline())

        # Yield events as they come from the queue
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    # End of stream
                    break

                # Track node results
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
            # Ensure pipeline task completes
            await pipeline_task

        # Yield final completion event
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
