"""Port call node for direct adapter method invocation.

This node allows calling any method on a configured port directly from YAML,
eliminating Python wrapper functions for simple adapter operations.

Examples
--------
Basic usage in Python::

    from hexdag.stdlib.nodes import PortCallNode

    node_factory = PortCallNode()
    node = node_factory(
        name="save_to_db",
        port="database",
        method="aexecute_query",
        input_mapping={"query": "$input.sql_query"}
    )

YAML pipeline usage::

    - kind: port_call_node
      metadata:
        name: execute_accept
      spec:
        port: database
        method: record_acceptance
        input_mapping:
          load_id: $input.load_id
          negotiation_id: get_context.negotiation.id
          carrier_id: get_context.carrier.id
      dependencies: [get_context]
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from hexdag.kernel.context import get_port
from hexdag.kernel.logging import get_logger
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from pydantic import BaseModel

    from hexdag.kernel.domain.dag import NodeSpec

logger = get_logger(__name__)


class PortCallNode(BaseNodeFactory, yaml_alias="port_call_node"):
    """Execute a method on a configured port/adapter.

    This node type eliminates the need for Python wrapper functions
    that simply extract fields from input and call adapter methods.
    The port, method, and parameter mapping are defined declaratively
    in the YAML configuration.

    The node:
    1. Resolves the port from the execution context
    2. Uses input_mapping to prepare method arguments (handled by orchestrator)
    3. Calls the method (supports both sync and async)
    4. Returns the result with metadata

    Parameters (in YAML spec)
    -------------------------
    port : str
        Name of the port to use (e.g., "database", "llm", "cache")
    method : str
        Method name to call on the port (e.g., "aexecute_query", "aget")
    input_mapping : dict[str, str], optional
        Mapping of method parameter names to data sources.
        Supports:
        - ``$input.field`` - Extract from initial pipeline input
        - ``dependency_name.field`` - Extract from a dependency's output
    fallback : Any, optional
        Value to return if the port is not available
    has_fallback : bool, optional
        Set to True to enable fallback behavior (allows None as fallback value)

    Examples
    --------
    >>> factory = PortCallNode()
    >>> node = factory(
    ...     name="call_db",
    ...     port="database",
    ...     method="aexecute_query",
    ... )
    >>> node.name
    'call_db'

    With input mapping::

        >>> node = factory(
        ...     name="record_data",
        ...     port="database",
        ...     method="record_acceptance",
        ...     input_mapping={
        ...         "load_id": "$input.load_id",
        ...         "carrier_id": "context.carrier.id",
        ...     },
        ...     deps=["context"],
        ... )
        >>> "context" in node.deps
        True
    """

    # Studio UI metadata
    _hexdag_icon = "Plug"
    _hexdag_color = "#84cc16"  # lime-500

    def __call__(
        self,
        name: str,
        port: str,
        method: str,
        input_mapping: dict[str, str] | None = None,
        fallback: Any = None,
        has_fallback: bool = False,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a port method invocation node.

        Parameters
        ----------
        name : str
            Node name (must be unique within the pipeline)
        port : str
            Name of the port to call (e.g., "database", "llm", "tool_router")
        method : str
            Method name to invoke on the port
        input_mapping : dict[str, str] | None, optional
            Mapping of method parameter names to data sources.
            Supports ``$input.field`` and ``dependency_name.field`` syntax.
        fallback : Any, optional
            Value to return if the port is not available
        has_fallback : bool, optional
            Set to True to enable fallback behavior (allows None as fallback)
        output_schema : dict[str, Any] | type[BaseModel] | None, optional
            Optional schema for validating/structuring the output
        deps : list[str] | None, optional
            List of dependency node names for execution ordering
        **kwargs : Any
            Additional parameters stored in NodeSpec.params

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution

        Examples
        --------
        >>> factory = PortCallNode()
        >>> node = factory(
        ...     name="save_to_db",
        ...     port="database",
        ...     method="aexecute_query",
        ... )
        >>> node.name
        'save_to_db'
        """
        # Capture configuration in closure
        port_name = port
        method_name = method
        _fallback = fallback
        _has_fallback = has_fallback

        async def port_call_fn(input_data: dict[str, Any]) -> dict[str, Any]:
            """Execute port method call."""
            node_logger = logger.bind(
                node=name,
                node_type="port_call_node",
                port=port_name,
                method=method_name,
            )

            # Get the port from context
            port_adapter = get_port(port_name)

            if port_adapter is None:
                if _has_fallback:
                    node_logger.warning("Port '{}' not available, using fallback", port_name)
                    return {
                        "result": _fallback,
                        "port": port_name,
                        "method": method_name,
                        "error": f"Port '{port_name}' not available",
                    }
                raise RuntimeError(
                    f"Port '{port_name}' not available in execution context. "
                    f"Ensure the port is configured in the orchestrator."
                )

            # Verify method exists
            if not hasattr(port_adapter, method_name):
                available = [m for m in dir(port_adapter) if not m.startswith("_")]
                raise AttributeError(
                    f"Port '{port_name}' has no method '{method_name}'. "
                    f"Available methods: {', '.join(available[:10])}"
                )

            method_fn = getattr(port_adapter, method_name)

            # Prepare method arguments from input_data
            # input_data is already processed by ExecutionCoordinator._apply_input_mapping
            # if input_mapping was specified in the node params
            method_kwargs = dict(input_data) if isinstance(input_data, dict) else {}

            node_logger.info(
                "Calling port method",
                args=list(method_kwargs.keys()),
            )

            try:
                # Call method (handle both sync and async)
                with node_timer() as t:
                    if asyncio.iscoroutinefunction(method_fn):
                        result = await method_fn(**method_kwargs)
                    else:
                        result = method_fn(**method_kwargs)

                node_logger.debug(
                    "Port method completed",
                    result_type=type(result).__name__,
                    duration_ms=t.duration_str,
                )

                return {
                    "result": result,
                    "port": port_name,
                    "method": method_name,
                    "error": None,
                }

            except Exception as e:
                node_logger.error(
                    "Port method failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )

                if _has_fallback:
                    return {
                        "result": _fallback,
                        "port": port_name,
                        "method": method_name,
                        "error": str(e),
                    }
                raise

        # Preserve function metadata for debugging
        port_call_fn.__name__ = f"port_call_{name}"
        port_call_fn.__doc__ = f"Port call: {port_name}.{method_name}"

        # Build input schema from input_mapping if provided
        input_schema = dict.fromkeys(input_mapping, Any) if input_mapping else None

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=port_call_fn,
            input_schema=input_schema,
            output_schema=output_schema,
            deps=deps,
            input_mapping=input_mapping,  # Pass to params for ExecutionCoordinator
            **kwargs,
        )
