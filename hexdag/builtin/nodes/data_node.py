"""Static data node for returning constant output.

This module provides a DataNode factory for creating nodes that return
static data without requiring Python functions. Useful for terminal
nodes like rejection actions or static configuration.

Examples
--------
Basic usage in Python::

    from hexdag.builtin.nodes import DataNode

    node_factory = DataNode()
    node = node_factory(
        name="reject_locked",
        output={"action": "REJECTED", "reason": "Load already has winner locked"}
    )

YAML pipeline usage::

    - kind: data_node
      metadata:
        name: reject_locked
      spec:
        output:
          action: "REJECTED"
          reason: "Load already has winner locked"
"""

from typing import Any

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger

from .base_node_factory import BaseNodeFactory

logger = get_logger(__name__)


class DataNode(BaseNodeFactory):
    """Static data node factory that returns constant output.

    This node type eliminates the need for trivial Python functions
    that simply return static dictionaries. The output is defined
    declaratively in the YAML configuration.

    The node ignores any input data and always returns the configured
    output. Dependencies can still be specified to control execution
    order in the DAG.

    Attributes
    ----------
    _yaml_schema : dict
        JSON Schema for YAML/MCP documentation

    Examples
    --------
    >>> factory = DataNode()
    >>> node = factory(
    ...     name="static_response",
    ...     output={"status": "OK", "code": 200}
    ... )
    >>> node.name
    'static_response'

    With dependencies::

        >>> node = factory(
        ...     name="after_validation",
        ...     output={"result": "validated"},
        ...     deps=["validator"]
        ... )
        >>> "validator" in node.deps
        True
    """

    _yaml_schema: dict[str, Any] = {
        "type": "object",
        "description": "Static data node returning constant output. "
        "Useful for terminal nodes like rejection actions or static configuration.",
        "properties": {
            "output": {
                "type": "object",
                "description": "Static output data to return. Can be any JSON-serializable object.",
            },
        },
        "required": ["output"],
    }

    def __call__(
        self,
        name: str,
        output: dict[str, Any],
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a static data node.

        Parameters
        ----------
        name : str
            Node name (must be unique within the pipeline)
        output : dict[str, Any]
            Static output data to return when the node executes
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
        >>> factory = DataNode()
        >>> node = factory(
        ...     name="reject_locked",
        ...     output={"action": "REJECTED", "reason": "Load locked"}
        ... )
        >>> node.name
        'reject_locked'
        """
        # Capture output in closure to avoid reference issues
        static_output = dict(output)

        async def data_fn(input_data: Any) -> dict[str, Any]:
            """Return static output, ignoring input data."""
            node_logger = logger.bind(node=name, node_type="data_node")

            _ = input_data  # Explicitly unused

            node_logger.debug(
                "Returning static data",
                output_keys=list(static_output.keys()),
                output_key_count=len(static_output),
            )

            return static_output

        # Preserve function metadata for debugging
        data_fn.__name__ = f"data_node_{name}"
        data_fn.__doc__ = f"Static data node returning: {static_output}"

        return NodeSpec(
            name=name,
            fn=data_fn,
            in_model=None,  # No input validation needed
            out_model=None,  # Output is dynamic based on YAML
            deps=frozenset(deps or []),
            params={"output": static_output, **kwargs},
        )
