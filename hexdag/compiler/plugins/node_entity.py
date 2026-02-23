"""Plugin for handling all regular node types (llm, function, agent, etc.)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.resolver import ResolveError, resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec


class NodeEntityPlugin:
    """Plugin for handling all node types (llm, function, agent, etc.)."""

    def __init__(self, builder: YamlPipelineBuilder):
        """Initialize with reference to builder for shared state."""
        self.builder = builder

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle everything except macro_invocation."""
        return node_config.get("kind") != "macro_invocation"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec:
        """Build node from config.

        The 'kind' field must be a full module path to the node factory class.
        Example: hexdag.stdlib.nodes.LLMNode
        """
        # Validate structure
        if "kind" not in node_config:
            raise YamlPipelineBuilderError("Node missing 'kind' field")
        if "metadata" not in node_config or "name" not in node_config["metadata"]:
            raise YamlPipelineBuilderError(
                f"Node '{node_config.get('kind')}' missing metadata.name"
            )

        kind = node_config["kind"]
        node_id = node_config["metadata"]["name"]
        spec = node_config.get("spec", {}).copy()
        # Dependencies can be at node level or inside spec (for backwards compatibility)
        deps = node_config.get("dependencies", []) or spec.pop("dependencies", [])
        # Snapshot factory params before factory consumes them
        factory_params_snapshot = spec.copy()

        # Resolve factory class from full module path
        try:
            factory_obj = resolve(kind)
        except ResolveError as e:
            raise YamlPipelineBuilderError(f"Cannot resolve node kind '{kind}': {e}") from e

        # Validate it's callable
        if not callable(factory_obj):
            raise YamlPipelineBuilderError(
                f"Node factory '{kind}' is not callable (got {type(factory_obj).__name__})"
            )

        # Handle factory classes vs factory functions
        # Factory classes need to be instantiated first, then called
        # Factory functions can be called directly
        if isinstance(factory_obj, type):
            # It's a class - instantiate then call
            factory_instance = factory_obj()
            factory = cast("Callable[..., NodeSpec]", factory_instance)
        else:
            # It's already a callable (function or instance)
            factory = cast("Callable[..., NodeSpec]", factory_obj)  # type: ignore[unreachable]

        # Create node - pass name as first positional arg
        node: NodeSpec = factory(node_id, **spec)

        # Store factory metadata for distributed execution (remote workers
        # can call resolve(kind)() and replay factory(name, **params))
        node = replace(node, factory_class=kind, factory_params=factory_params_snapshot)

        # Add dependencies
        if deps:
            node = node.after(*deps) if isinstance(deps, list) else node.after(deps)

        return node
