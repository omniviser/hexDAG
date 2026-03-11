"""Plugin for handling Macro definitions (kind: Macro)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import register_runtime

if TYPE_CHECKING:
    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)


class MacroDefinitionPlugin:
    """Plugin for handling Macro definitions (kind: Macro).

    This plugin processes YAML macro definitions and registers them in the
    component registry for later invocation. Macro definitions don't add
    nodes to the graph - they just register reusable templates.

    Examples
    --------
    YAML macro definition::

        apiVersion: hexdag/v1
        kind: Macro
        metadata:
          name: retry_workflow
          description: Retry logic with exponential backoff
        parameters:
          - name: max_retries
            type: int
            default: 3
        nodes:
          - kind: function_node
            metadata:
              name: "{{name}}_attempt"
            spec:
              fn: "{{fn}}"
    """

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle Macro kind."""
        return node_config.get("kind") == "Macro"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Register YAML macro in component registry.

        Parameters
        ----------
        node_config : dict[str, Any]
            Macro definition configuration
        builder : YamlPipelineBuilder
            Builder instance (unused)
        graph : DirectedGraph
            Graph instance (unused - definitions don't add nodes)

        Returns
        -------
        None
            Macro definitions don't add nodes to the graph
        """
        from hexdag.kernel.yaml_macro import (  # lazy: mutual cycle with yaml_builder
            PortRequirement,
            YamlMacro,
            YamlMacroConfig,
            YamlMacroParameterSpec,
        )

        # Extract metadata
        metadata = node_config.get("metadata", {})
        macro_name = metadata.get("name")
        if not macro_name:
            raise YamlPipelineBuilderError("Macro definition missing 'metadata.name'")

        macro_description = metadata.get("description")
        _ = metadata.get("namespace", "user")  # Reserved for future namespace support

        # Extract parameters
        raw_parameters = node_config.get("parameters", [])
        parameters = [YamlMacroParameterSpec(**p) for p in raw_parameters]

        # Extract nodes — support both static list and raw YAML template string
        nodes = node_config.get("nodes", [])
        nodes_raw = node_config.get("nodes_raw")

        if not nodes and not nodes_raw:
            raise YamlPipelineBuilderError(
                f"Macro '{macro_name}' has no nodes. "
                f"Macros must define 'nodes' (list) or 'nodes_raw' (template string)."
            )

        # Extract outputs (optional)
        outputs = node_config.get("outputs")

        # Extract port requirements (optional)
        raw_ports = node_config.get("requires_ports", [])
        requires_ports = [PortRequirement(**p) for p in raw_ports]

        # Create YamlMacroConfig
        macro_config = YamlMacroConfig(
            macro_name=macro_name,
            macro_description=macro_description,
            parameters=parameters,
            nodes=nodes,
            nodes_raw=nodes_raw,
            requires_ports=requires_ports,
            outputs=outputs,
        )

        # Register macro at runtime
        # Create a dynamic class that pre-fills the YamlMacro config
        config_dict = macro_config.model_dump()

        # Structural fields that must not be overridden at invocation time
        _structural = frozenset({
            "macro_name",
            "macro_description",
            "parameters",
            "nodes",
            "nodes_raw",
            "requires_ports",
            "outputs",
        })

        class DynamicYamlMacro(YamlMacro):
            """Dynamically generated YamlMacro with pre-filled configuration."""

            def __init__(self, **kwargs: Any) -> None:
                # Only allow non-structural overrides (parameter values)
                safe_kwargs = {k: v for k, v in kwargs.items() if k not in _structural}
                merged_config = {**config_dict, **safe_kwargs}
                super().__init__(**merged_config)

        # Set class name for better debugging
        DynamicYamlMacro.__name__ = f"YamlMacro_{macro_name}"
        DynamicYamlMacro.__qualname__ = f"YamlMacro_{macro_name}"

        # Register in runtime storage (for YAML-defined macros)
        register_runtime(macro_name, DynamicYamlMacro)

        logger.info(
            "Registered YAML macro '{}' ({} parameters, {} nodes)",
            macro_name,
            len(parameters),
            len(nodes),
        )

        # Return None - macro definitions don't add nodes to the graph
        return None
