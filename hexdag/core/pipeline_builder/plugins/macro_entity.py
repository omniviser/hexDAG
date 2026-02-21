"""Plugin for handling macro invocation entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.core.configurable import ConfigurableMacro
from hexdag.core.logging import get_logger
from hexdag.core.resolver import ResolveError, resolve

if TYPE_CHECKING:
    from hexdag.core.domain.dag import DirectedGraph, NodeSpec
    from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

logger = get_logger(__name__)


class MacroEntityPlugin:
    """Plugin for handling macro_invocation entities."""

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle macro_invocation kind."""
        return node_config.get("kind") == "macro_invocation"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Expand macro into subgraph and merge into main graph."""
        from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilderError

        instance_name = node_config["metadata"]["name"]
        spec = node_config.get("spec", {})
        macro_ref = spec.get("macro")
        if not macro_ref:
            raise YamlPipelineBuilderError(f"Macro '{instance_name}' missing spec.macro field")

        # macro_ref is the full module path (e.g., hexdag.builtin.macros.ReasoningAgentMacro)
        # or a runtime-registered name for YAML-defined macros

        # Get config params for macro initialization
        config_params = spec.get("config", {}).copy()
        inputs = spec.get("inputs", {})
        dependencies = spec.get("dependencies", [])

        # Resolve macro class - either full module path or runtime-registered name
        try:
            macro_cls = resolve(macro_ref)
        except ResolveError as e:
            raise YamlPipelineBuilderError(f"Macro '{macro_ref}' not found: {e}") from e

        # Instantiate macro with config params
        try:
            macro_instance_obj = macro_cls(**config_params)
        except Exception as e:
            raise YamlPipelineBuilderError(f"Failed to instantiate macro '{macro_ref}': {e}") from e

        # Validate it's actually a macro
        if not isinstance(macro_instance_obj, ConfigurableMacro):
            type_name = type(macro_instance_obj).__name__
            raise YamlPipelineBuilderError(
                f"Component '{macro_ref}' is not a ConfigurableMacro (got {type_name})"
            )

        macro_instance: ConfigurableMacro = macro_instance_obj

        # Expand macro - merge config and inputs for validation
        # config params are the parameter values for the macro
        # inputs are the value mappings, so they should be merged
        macro_inputs = {**config_params, **inputs}

        # Expand macro
        try:
            subgraph = macro_instance.expand(
                instance_name=instance_name, inputs=macro_inputs, dependencies=dependencies
            )
        except ValueError:
            # Re-raise validation errors directly (e.g., required parameter, enum validation)
            raise
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to expand macro '{macro_ref}' (instance '{instance_name}'): {e}"
            ) from e

        # Merge subgraph into main graph
        self._merge_subgraph(graph, subgraph, dependencies)

        logger.info(
            "Expanded macro '{macro}' as '{instance}' ({nodes} nodes)",
            macro=macro_ref,
            instance=instance_name,
            nodes=len(subgraph.nodes),
        )

        # Return None - subgraph already merged into graph
        return None

    @staticmethod
    def _merge_subgraph(
        graph: DirectedGraph, subgraph: DirectedGraph, external_deps: list[str]
    ) -> None:
        """Merge subgraph into main graph with external dependencies.

        Optimized to avoid unnecessary graph copies when no external dependencies exist.
        """
        if not external_deps:
            # Fast path: direct merge when no external dependencies
            graph |= subgraph
        else:
            # Only process entry nodes that need external dependencies
            # Use in-place merge for better performance
            for node in subgraph.nodes.values():
                if not subgraph.get_dependencies(node.name):
                    # Entry node - add external dependencies
                    graph += node.after(*external_deps)
                else:
                    # Internal node - add as-is
                    graph += node
