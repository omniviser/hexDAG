"""Plugin for handling Adapter definitions (kind: Adapter).

Standalone ``kind: Adapter`` manifests define reusable adapter configurations
that can be referenced by name from ``spec.ports.<name>.ref``.

Example YAML::

    apiVersion: hexdag/v1
    kind: Adapter
    metadata:
      name: production-openai
      description: Production OpenAI with low temperature
    spec:
      class: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4o
        temperature: 0.3
        api_key: ${OPENAI_API_KEY}

Reference in a pipeline::

    spec:
      ports:
        llm:
          ref: production-openai
          config:          # optional overrides merged on top
            temperature: 0.9
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)

# Module-level registry for named adapter definitions
_adapter_registry: dict[str, dict[str, Any]] = {}


def get_adapter_definition(name: str) -> dict[str, Any] | None:
    """Look up a registered adapter definition by name.

    Parameters
    ----------
    name : str
        Adapter name (from ``metadata.name``)

    Returns
    -------
    dict[str, Any] | None
        Adapter definition dict with ``class`` and ``config`` keys,
        or None if not registered
    """
    return _adapter_registry.get(name)


def clear_adapter_registry() -> None:
    """Clear all registered adapter definitions (for testing)."""
    _adapter_registry.clear()


class AdapterDefinitionPlugin:
    """Plugin for handling Adapter definitions (kind: Adapter).

    Processes ``kind: Adapter`` YAML documents and registers the
    adapter configuration for later reference by name.  Does not add
    nodes to the graph.
    """

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle Adapter kind."""
        return node_config.get("kind") == "Adapter"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Register named adapter definition.

        Parameters
        ----------
        node_config : dict[str, Any]
            Adapter definition configuration
        builder : YamlPipelineBuilder
            Builder instance (unused)
        graph : DirectedGraph
            Graph instance (unused — definitions don't add nodes)

        Returns
        -------
        None
            Adapter definitions don't add nodes to the graph
        """
        metadata = node_config.get("metadata", {})
        name = metadata.get("name")
        if not name:
            raise YamlPipelineBuilderError("Adapter definition missing 'metadata.name'")

        spec = node_config.get("spec", {})
        adapter_class = spec.get("class")

        if not adapter_class:
            raise YamlPipelineBuilderError(
                f"Adapter '{name}' missing 'spec.class'. "
                f"Adapter definitions must specify the adapter class module path."
            )

        if not isinstance(adapter_class, str):
            raise YamlPipelineBuilderError(
                f"Adapter '{name}' spec.class must be a module path string, "
                f"got {type(adapter_class).__name__}"
            )

        config = spec.get("config", {})
        if not isinstance(config, dict):
            raise YamlPipelineBuilderError(
                f"Adapter '{name}' spec.config must be a dict, got {type(config).__name__}"
            )

        capabilities = spec.get("capabilities")

        entry: dict[str, Any] = {"class": adapter_class, "config": config}
        if capabilities is not None:
            entry["capabilities"] = capabilities

        _adapter_registry[name] = entry

        logger.info(
            "Registered adapter definition '{}' (class={})",
            name,
            adapter_class,
        )

        return None
