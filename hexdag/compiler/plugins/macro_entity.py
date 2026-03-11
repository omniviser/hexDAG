"""Plugin for handling macro invocation entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.compiler.plugins.node_entity import NodeEntityPlugin
from hexdag.kernel.configurable import ConfigurableMacro
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import ResolveError, resolve
from hexdag.kernel.yaml_macro import YamlMacro, YamlMacroConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexdag.compiler.yaml_builder import EntityPlugin, YamlPipelineBuilder
    from hexdag.kernel.domain.dag import NodeSpec

logger = get_logger(__name__)


class MacroEntityPlugin:
    """Plugin for handling macro_invocation entities."""

    def __init__(self) -> None:
        # Exit nodes of the last expanded macro, used by the builder
        # to set correct implicit dependencies for the next node.
        self.last_exit_nodes: list[str] = []

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle macro_invocation kind."""
        return node_config.get("kind") == "macro_invocation"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Expand macro into subgraph and merge into main graph."""
        instance_name = node_config["metadata"]["name"]
        spec = node_config.get("spec", {})
        macro_ref = spec.get("macro")
        if not macro_ref:
            raise YamlPipelineBuilderError(f"Macro '{instance_name}' missing spec.macro field")

        # macro_ref is the full module path (e.g., hexdag.stdlib.macros.ReasoningAgentMacro)
        # or a runtime-registered name for YAML-defined macros

        # Get config params for macro initialization and inputs for expansion.
        # config → macro __init__ (class-level settings)
        # inputs → macro expand() (instance-level values)
        config_params = spec.get("config", {}).copy()
        inputs = spec.get("inputs", {})
        dependencies = spec.get("dependencies", [])

        # Reject overlapping keys — forces users to be explicit about intent
        overlap = set(config_params) & set(inputs)
        if overlap:
            raise YamlPipelineBuilderError(
                f"Macro '{instance_name}': keys {sorted(overlap)} appear in both "
                f"'config' and 'inputs'. Use 'config' for macro initialization, "
                f"'inputs' for expansion parameters."
            )

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

        # Validate port requirements (YAML macros only)
        self._validate_port_requirements(macro_instance, instance_name, macro_ref, builder)

        # Expand macro — only pass inputs (not config) to expand().
        # Config was already consumed by __init__; passing it again would
        # double-apply values and silently override inputs on collision.
        # For YamlMacro, inject node_builder callback to avoid circular
        # import (yaml_macro → compiler → yaml_macro).
        # The expansion_guard detects circular A→B→A references and
        # enforces a depth limit.
        try:
            with macro_instance.expansion_guard(instance_name):
                if isinstance(macro_instance, YamlMacro):
                    subgraph = macro_instance.expand(
                        instance_name=instance_name,
                        inputs=inputs,
                        dependencies=dependencies,
                        node_builder=self._make_node_builder(builder),
                    )
                else:
                    subgraph = macro_instance.expand(
                        instance_name=instance_name,
                        inputs=inputs,
                        dependencies=dependencies,
                    )
        except (ValueError, YamlPipelineBuilderError):
            # Re-raise validation/builder errors directly
            raise
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to expand macro '{macro_ref}' (instance '{instance_name}'): {e}"
            ) from e

        # Track exit nodes BEFORE merging (get_exit_nodes checks the
        # subgraph in isolation, which is what we want).
        self.last_exit_nodes = subgraph.get_exit_nodes()

        # Merge subgraph into main graph
        self._merge_subgraph(graph, subgraph, dependencies)

        logger.info(
            "Expanded macro '{macro}' as '{instance}' ({nodes} nodes)",
            macro=macro_ref,
            instance=instance_name,
            nodes=len(subgraph),
        )

        # Return None - subgraph already merged into graph
        return None

    @staticmethod
    def _validate_port_requirements(
        macro_instance: ConfigurableMacro,
        instance_name: str,
        macro_ref: str,
        builder: YamlPipelineBuilder,
    ) -> None:
        """Validate that the pipeline provides all ports required by the macro.

        Only applies to YAML macros with ``requires_ports`` declared.
        """
        config = getattr(macro_instance, "config", None)
        if not isinstance(config, YamlMacroConfig):
            return
        if not config.requires_ports:
            return

        pipeline_ports = getattr(builder, "pipeline_ports", {})
        for req in config.requires_ports:
            if not req.optional and req.name not in pipeline_ports:
                raise YamlPipelineBuilderError(
                    f"Macro '{macro_ref}' (instance '{instance_name}') requires "
                    f"port '{req.name}' ({req.protocol}) but it is not declared "
                    f"in pipeline spec.ports"
                )

    @staticmethod
    def _make_node_builder(
        builder: YamlPipelineBuilder,
    ) -> Callable[[list[dict[str, Any]]], DirectedGraph]:
        """Create a node-builder callback from the current YamlPipelineBuilder.

        The callback builds a DirectedGraph from a list of rendered node configs
        using the builder's entity plugins — this is injected into YamlMacro to
        avoid the circular import between yaml_macro and the compiler.

        Supports both regular nodes AND nested macro_invocation nodes,
        enabling YAML macro composition.
        """
        node_plugin = NodeEntityPlugin(builder)
        macro_plugin = MacroEntityPlugin()
        plugins: list[EntityPlugin] = [macro_plugin, node_plugin]

        def _build(rendered_nodes: list[dict[str, Any]]) -> DirectedGraph:
            graph = DirectedGraph()
            for node_config in rendered_nodes:
                handled = False
                for plugin in plugins:
                    if plugin.can_handle(node_config):
                        result = plugin.build(node_config, builder, graph)
                        if result is not None:
                            graph += result
                        handled = True
                        break
                if not handled:
                    kind = node_config.get("kind", "unknown")
                    raise ValueError(f"Invalid node kind in YAML macro: {kind}")
            return graph

        return _build

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
            # Collect entry nodes BEFORE any mutation to avoid order-dependent detection
            entry_nodes = frozenset(
                node.name for node in subgraph if not subgraph.get_dependencies(node.name)
            )
            for node in subgraph:
                if node.name in entry_nodes:
                    # Entry node - add external dependencies
                    graph += node.after(*external_deps)
                else:
                    # Internal node - add as-is
                    graph += node
