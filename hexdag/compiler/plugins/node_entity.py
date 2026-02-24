"""Plugin for handling all regular node types (llm, function, agent, etc.)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.resolver import ResolveError, resolve
from hexdag.kernel.schema.generator import SchemaGenerator

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

        # Read settings (literal config) and spec (dynamic wiring) sections
        settings_values = node_config.get("settings", {}).copy()
        spec = node_config.get("spec", {}).copy()

        # Dependencies can be at node level or inside spec (for backwards compatibility)
        deps = node_config.get("dependencies", []) or spec.pop("dependencies", [])

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

        # Backward compat: if no explicit settings section, auto-partition flat spec
        if not settings_values:
            settings_values, spec = self._partition_spec(factory_obj, spec)

        # Merge for factory call (factory signature unchanged)
        factory_kwargs = {**settings_values, **spec}

        # Snapshot factory params before factory consumes them
        factory_params_snapshot = factory_kwargs.copy()

        # Create node - pass name as first positional arg
        node: NodeSpec = factory(node_id, **factory_kwargs)

        # Store factory metadata and literals on NodeSpec
        node = replace(
            node,
            factory_class=kind,
            factory_params=factory_params_snapshot,
            literals=settings_values,
        )

        # Add dependencies
        if deps:
            node = node.after(*deps) if isinstance(deps, list) else node.after(deps)

        return node

    @staticmethod
    def _partition_spec(
        factory_obj: Any, spec: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition a flat spec dict into (literals, dynamic_params).

        Uses ``SchemaGenerator.get_literal_param_names()`` to determine which
        params are literal settings vs dynamic wiring.

        Parameters
        ----------
        factory_obj : Any
            Resolved factory class or instance
        spec : dict[str, Any]
            Flat spec dict to partition

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            (settings_values, remaining_spec)
        """
        literal_names = SchemaGenerator.get_literal_param_names(factory_obj)

        settings = {k: v for k, v in spec.items() if k in literal_names}
        remaining = {k: v for k, v in spec.items() if k not in literal_names}

        return settings, remaining
