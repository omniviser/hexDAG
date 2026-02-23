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

        # Extract nodes
        nodes = node_config.get("nodes", [])
        if not nodes:
            raise YamlPipelineBuilderError(
                f"Macro '{macro_name}' has no nodes. Macros must define at least one node."
            )

        # Extract outputs (optional)
        outputs = node_config.get("outputs")

        # Create YamlMacroConfig
        macro_config = YamlMacroConfig(
            macro_name=macro_name,
            macro_description=macro_description,
            parameters=parameters,
            nodes=nodes,
            outputs=outputs,
        )

        # Register macro at runtime
        # Create a dynamic class that pre-fills the YamlMacro config
        config_dict = macro_config.model_dump()

        class DynamicYamlMacro(YamlMacro):
            """Dynamically generated YamlMacro with pre-filled configuration."""

            def __init__(self, **kwargs: Any) -> None:
                # Merge pre-filled config with any override kwargs
                merged_config = {**config_dict, **kwargs}
                super().__init__(**merged_config)

        # Set class name for better debugging
        DynamicYamlMacro.__name__ = f"YamlMacro_{macro_name}"
        DynamicYamlMacro.__qualname__ = f"YamlMacro_{macro_name}"

        # Register in runtime storage (for YAML-defined macros)
        register_runtime(macro_name, DynamicYamlMacro)

        logger.info(
            f"Registered YAML macro '{macro_name}' "
            f"({len(parameters)} parameters, {len(nodes)} nodes)"
        )

        # Return None - macro definitions don't add nodes to the graph
        return None
