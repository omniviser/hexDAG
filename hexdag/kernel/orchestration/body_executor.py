"""BodyExecutor - Shared body execution logic for ControlFlowNode.

This module provides unified body execution for control flow nodes,
supporting four execution modes:
1. Module path string - Resolve and call function by path
2. !py inline Python - Compile and execute inline function
3. Inline nodes - Build and execute sub-DAG
4. Pipeline reference - Load and execute external pipeline

Examples
--------
Basic usage::

    executor = BodyExecutor()

    # Execute function by module path
    result = await executor.execute(
        body="myapp.process_item",
        input_data={"item": data},
        context=ctx,
        ports={"database": db_adapter},
    )

    # Execute inline nodes
    result = await executor.execute(
        body=[
            {"kind": "expression_node", "metadata": {"name": "step1"}, "spec": {...}},
            {"kind": "llm_node", "metadata": {"name": "step2"}, "spec": {...}},
        ],
        input_data={"item": data},
        context=ctx,
        ports={"llm": llm_adapter},
    )
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hexdag.kernel.context import get_user_ports
from hexdag.kernel.exceptions import BodyExecutorError  # noqa: F401
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.models import NodeExecutionContext
from hexdag.kernel.resolver import resolve_function

logger = get_logger(__name__)


class BodyExecutor:
    """Execute node body in function, inline-nodes, !py, or pipeline mode.

    This class provides a unified interface for executing different body types
    within control flow nodes (while, for-each, times, if-else, switch).

    The BodyExecutor can be configured with orchestrator settings to ensure
    sub-DAGs and pipelines execute with consistent configuration.

    Attributes
    ----------
    base_path : Path
        Base directory for resolving pipeline references
    max_concurrent_nodes : int
        Maximum concurrent nodes for sub-orchestrators
    strict_validation : bool
        Whether to use strict validation in sub-orchestrators
    default_node_timeout : float | None
        Default timeout for nodes in sub-orchestrators

    Examples
    --------
    Execute a function by module path::

        executor = BodyExecutor()
        result = await executor.execute(
            body="myapp.process",
            input_data={"x": 1},
            context=ctx,
            ports={},
        )

    Execute inline Python function (from !py tag)::

        async def my_func(item, index, state, **ports):
            return item * 2

        result = await executor.execute(
            body=my_func,  # Compiled from !py tag
            input_data={"item": 5, "index": 0, "state": {}},
            context=ctx,
            ports={},
        )

    With orchestrator configuration::

        executor = BodyExecutor(
            base_path=Path("./pipelines"),
            max_concurrent_nodes=5,
            strict_validation=True,
        )
    """

    def __init__(
        self,
        base_path: Path | None = None,
        max_concurrent_nodes: int = 10,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
    ) -> None:
        """Initialize BodyExecutor.

        Parameters
        ----------
        base_path : Path | None
            Base directory for resolving pipeline references.
            Defaults to current working directory.
        max_concurrent_nodes : int
            Maximum concurrent nodes for sub-orchestrators (default: 10)
        strict_validation : bool
            Whether to use strict validation in sub-orchestrators (default: False)
        default_node_timeout : float | None
            Default timeout for nodes in sub-orchestrators (default: None)
        """
        self.base_path = base_path or Path.cwd()
        self.max_concurrent_nodes = max_concurrent_nodes
        self.strict_validation = strict_validation
        self.default_node_timeout = default_node_timeout

    async def execute(
        self,
        body: str | list[dict[str, Any]] | Callable[..., Any] | None,
        body_pipeline: str | None,
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
        iteration_context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute body and return result.

        Dispatches to the appropriate execution method based on body type:
        - str: Module path string → _execute_function
        - Callable: !py compiled function → _execute_py_function
        - list[dict]: Inline nodes → _execute_inline_nodes
        - body_pipeline: Pipeline reference → _execute_pipeline

        Parameters
        ----------
        body : str | list[dict] | Callable | None
            The body specification. Can be:
            - Module path string (e.g., "myapp.process")
            - Callable (compiled from !py tag)
            - List of node configs (inline nodes/sub-DAG)
        body_pipeline : str | None
            Path to external pipeline YAML file
        input_data : dict[str, Any]
            Input data for the body execution
        context : NodeExecutionContext
            Execution context with metadata
        ports : dict[str, Any]
            Injected ports (llm, database, memory, etc.)
        iteration_context : dict[str, Any] | None
            Additional context for iterations ($item, $index, $total, state)

        Returns
        -------
        Any
            Result from body execution

        Raises
        ------
        BodyExecutorError
            If body execution fails or no body is specified
        """
        # Merge iteration context with input data
        exec_context = dict(input_data)
        if iteration_context:
            exec_context.update(iteration_context)

        # Fall back to ContextVar ports when caller passed empty dict
        # (e.g., CompositeNode receives empty **ports from NodeExecutor)
        if not ports:
            ports = get_user_ports()

        if body_pipeline:
            return await self._execute_pipeline(body_pipeline, exec_context, context, ports)
        if isinstance(body, list):
            return await self._execute_inline_nodes(body, exec_context, context, ports)
        if isinstance(body, str):
            return await self._execute_function(body, exec_context, context, ports)
        if callable(body):
            # Callable passed directly (either !py compiled function or regular callable)
            # We call with (input_data, **ports) to match function_node convention
            return await self._execute_callable(body, exec_context, context, ports)
        raise BodyExecutorError(
            "No body specified. Provide 'body' (function path, callable, or inline nodes) "
            "or 'body_pipeline' (pipeline reference)."
        )

    async def _execute_function(
        self,
        body: str,
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute body as module path function.

        Parameters
        ----------
        body : str
            Module path to function (e.g., "myapp.process")
        input_data : dict[str, Any]
            Input data to pass to function
        context : NodeExecutionContext
            Execution context
        ports : dict[str, Any]
            Injected ports

        Returns
        -------
        Any
            Function result
        """
        try:
            func = resolve_function(body)
        except Exception as e:
            raise BodyExecutorError(f"Failed to resolve function '{body}': {e}") from e

        logger.debug(
            "Executing function body",
            function=body,
            node_id=context.node_id,
        )

        try:
            # Call function with input_data and ports
            # Support both sync and async functions
            result = func(input_data, **ports)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            raise BodyExecutorError(f"Function '{body}' execution failed: {e}") from e

    async def _execute_py_function(
        self,
        body: Callable[..., Any],
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute !py compiled inline Python function.

        The function is expected to have signature:
            async def process(item, index, state, **ports) -> Any

        But we support flexible signatures - pass what's available.

        Parameters
        ----------
        body : Callable
            Compiled Python function from !py tag
        input_data : dict[str, Any]
            Input data containing item, index, state, etc.
        context : NodeExecutionContext
            Execution context
        ports : dict[str, Any]
            Injected ports

        Returns
        -------
        Any
            Function result
        """
        logger.debug(
            "Executing !py inline function",
            function_name=getattr(body, "__name__", "<anonymous>"),
            node_id=context.node_id,
        )

        try:
            # Extract standard parameters from input_data
            item = input_data.get("$item", input_data.get("item"))
            index = input_data.get("$index", input_data.get("index", 0))
            state = input_data.get("state", {})

            # Call with positional args (item, index, state) and keyword args (ports)
            result = body(item, index, state, **ports)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            func_name = getattr(body, "__name__", "<anonymous>")
            raise BodyExecutorError(f"!py function '{func_name}' execution failed: {e}") from e

    async def _execute_callable(
        self,
        body: Callable[..., Any],
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute a callable body with function_node-style signature.

        This method handles regular callable bodies passed directly to
        ControlFlowNode. It uses the (input_data, **ports) convention
        matching function_node behavior.

        For !py compiled functions that need (item, index, state, **ports),
        use _execute_py_function instead.

        Parameters
        ----------
        body : Callable
            A callable function
        input_data : dict[str, Any]
            Input data to pass as first argument
        context : NodeExecutionContext
            Execution context
        ports : dict[str, Any]
            Injected ports

        Returns
        -------
        Any
            Function result
        """
        logger.debug(
            "Executing callable body",
            function_name=getattr(body, "__name__", "<anonymous>"),
            node_id=context.node_id,
        )

        try:
            # Call with (input_data, **ports) - matching function_node convention
            result = body(input_data, **ports)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            func_name = getattr(body, "__name__", "<anonymous>")
            raise BodyExecutorError(f"Callable '{func_name}' execution failed: {e}") from e

    async def _execute_inline_nodes(
        self,
        body: list[dict[str, Any]],
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute inline nodes as sub-DAG.

        Builds a temporary DirectedGraph from the inline node configs,
        creates a sub-orchestrator, and executes it.

        Parameters
        ----------
        body : list[dict[str, Any]]
            List of node configuration dicts
        input_data : dict[str, Any]
            Input data for the sub-DAG
        context : NodeExecutionContext
            Execution context
        ports : dict[str, Any]
            Injected ports

        Returns
        -------
        Any
            Result from the last non-skipped node in the sub-DAG
        """
        from hexdag.compiler import YamlPipelineBuilder  # lazy: cycle via composite_node
        from hexdag.kernel.domain.dag import DirectedGraph  # lazy: cycle via composite_node
        from hexdag.kernel.orchestration.orchestrator import (
            Orchestrator,  # lazy: cycle via __init__
        )

        logger.debug(
            "Executing inline nodes",
            node_count=len(body),
            node_id=context.node_id,
        )

        # Build sub-graph from inline nodes
        builder = YamlPipelineBuilder(base_path=self.base_path)
        sub_graph = DirectedGraph()

        for node_config in body:
            # Find appropriate plugin to build node
            for plugin in builder.entity_plugins:
                if plugin.can_handle(node_config):
                    result = plugin.build(node_config, builder, sub_graph)
                    if result is not None:
                        sub_graph += result
                    break

        if not sub_graph:
            logger.warning("Inline body has no nodes", node_id=context.node_id)
            return None

        # Create sub-orchestrator with inherited configuration
        sub_orchestrator = Orchestrator(
            max_concurrent_nodes=self.max_concurrent_nodes,
            strict_validation=self.strict_validation,
            default_node_timeout=self.default_node_timeout,
            ports=ports,
        )

        try:
            run_result = await sub_orchestrator.run(sub_graph, input_data, validate=False)
        except Exception as e:
            raise BodyExecutorError(f"Inline nodes execution failed: {e}") from e

        # Return the result from the last non-skipped node
        # Find nodes in topological order and get last result
        # Note: orchestrator.run() returns dict[str, Any]
        for node_name in reversed(list(sub_graph.keys())):
            if node_name in run_result:
                node_result = run_result[node_name]
                if isinstance(node_result, dict) and node_result.get("_skipped"):
                    continue
                return node_result

        return run_result

    async def _execute_pipeline(
        self,
        body_pipeline: str,
        input_data: dict[str, Any],
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute external pipeline from YAML file.

        Parameters
        ----------
        body_pipeline : str
            Path to pipeline YAML file (relative to base_path)
        input_data : dict[str, Any]
            Input data for the pipeline
        context : NodeExecutionContext
            Execution context
        ports : dict[str, Any]
            Injected ports

        Returns
        -------
        Any
            Pipeline execution result
        """
        from hexdag.compiler import YamlPipelineBuilder  # lazy: cycle via composite_node
        from hexdag.kernel.orchestration.orchestrator import (
            Orchestrator,  # lazy: cycle via __init__
        )

        # Resolve pipeline path
        pipeline_path = self.base_path / body_pipeline
        if not pipeline_path.exists():
            raise BodyExecutorError(f"Pipeline file not found: {pipeline_path}")

        logger.debug(
            "Executing pipeline body",
            pipeline=body_pipeline,
            node_id=context.node_id,
        )

        try:
            # Build pipeline
            builder = YamlPipelineBuilder(base_path=pipeline_path.parent)
            graph, config = builder.build_from_yaml_file(str(pipeline_path))

            # Create orchestrator with inherited configuration
            orchestrator = Orchestrator(
                max_concurrent_nodes=self.max_concurrent_nodes,
                strict_validation=self.strict_validation,
                default_node_timeout=self.default_node_timeout,
                ports=ports,
            )

            # Execute pipeline
            return await orchestrator.run(graph, input_data, validate=False)

        except Exception as e:
            raise BodyExecutorError(f"Pipeline '{body_pipeline}' execution failed: {e}") from e
