"""Plugin for handling Middleware definitions (kind: Middleware).

Standalone ``kind: Middleware`` manifests define reusable middleware stacks
that can be referenced by name from ``spec.ports.<name>.middleware``.

Example YAML::

    apiVersion: hexdag/v1
    kind: Middleware
    metadata:
      name: production-llm
    spec:
      stack:
        - hexdag.stdlib.middleware.RetryMiddleware
        - hexdag.stdlib.middleware.RateLimiter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)

# Module-level registry for named middleware stacks
_middleware_registry: dict[str, list[str]] = {}


def get_middleware_stack(name: str) -> list[str] | None:
    """Look up a registered middleware stack by name.

    Parameters
    ----------
    name : str
        Middleware stack name (from ``metadata.name``)

    Returns
    -------
    list[str] | None
        List of middleware module paths, or None if not registered
    """
    return _middleware_registry.get(name)


def clear_middleware_registry() -> None:
    """Clear all registered middleware stacks (for testing)."""
    _middleware_registry.clear()


class MiddlewareDefinitionPlugin:
    """Plugin for handling Middleware definitions (kind: Middleware).

    Processes ``kind: Middleware`` YAML documents and registers the
    middleware stack for later reference by name.  Does not add nodes
    to the graph.
    """

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle Middleware kind."""
        return node_config.get("kind") == "Middleware"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Register named middleware stack.

        Parameters
        ----------
        node_config : dict[str, Any]
            Middleware definition configuration
        builder : YamlPipelineBuilder
            Builder instance (unused)
        graph : DirectedGraph
            Graph instance (unused — definitions don't add nodes)

        Returns
        -------
        None
            Middleware definitions don't add nodes to the graph
        """
        metadata = node_config.get("metadata", {})
        name = metadata.get("name")
        if not name:
            raise YamlPipelineBuilderError("Middleware definition missing 'metadata.name'")

        spec = node_config.get("spec", {})
        stack = spec.get("stack", [])

        if not stack:
            raise YamlPipelineBuilderError(
                f"Middleware '{name}' has no stack. "
                f"Middleware definitions must have 'spec.stack' (list of module paths)."
            )

        if not isinstance(stack, list):
            raise YamlPipelineBuilderError(
                f"Middleware '{name}' stack must be a list, got {type(stack).__name__}"
            )

        # Validate all entries are strings (module paths)
        for i, entry in enumerate(stack):
            if not isinstance(entry, str):
                raise YamlPipelineBuilderError(
                    f"Middleware '{name}' stack[{i}] must be a module path string, "
                    f"got {type(entry).__name__}"
                )

        _middleware_registry[name] = stack

        logger.info(
            "Registered middleware stack '{}' ({} layers)",
            name,
            len(stack),
        )

        return None
