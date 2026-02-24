"""Plugin for handling Config definitions (kind: Config).

Processes ``kind: Config`` documents that appear in multi-document YAML
alongside Pipeline definitions. Extracts the ``spec`` section and produces
a ``HexDAGConfig``, storing it on the builder for downstream use.

This is a document-level plugin (like ``MacroDefinitionPlugin``).
Config definitions do not add nodes to the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.compiler.config_loader import ConfigLoader
from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

logger = get_logger(__name__)


class ConfigDefinitionPlugin:
    """Plugin for handling Config definitions (kind: Config).

    Processes ``kind: Config`` documents in multi-document YAML and converts
    them to ``HexDAGConfig`` objects stored on the builder instance.
    """

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle Config kind."""
        return node_config.get("kind") == "Config"

    def build(
        self,
        node_config: dict[str, Any],
        builder: YamlPipelineBuilder,
        graph: DirectedGraph,
    ) -> NodeSpec | None:
        """Parse Config document and store on builder.

        Parameters
        ----------
        node_config : dict[str, Any]
            The full ``kind: Config`` document
        builder : YamlPipelineBuilder
            Builder instance — stores parsed config on ``builder._inline_config``
        graph : DirectedGraph
            Graph instance (unused)

        Returns
        -------
        None
            Config definitions don't add nodes to the graph
        """
        metadata = node_config.get("metadata", {})
        config_name = metadata.get("name", "inline-config")
        spec = node_config.get("spec", {})

        loader = ConfigLoader()
        spec = loader._substitute_env_vars(spec)
        config = loader._parse_config(spec)

        builder._inline_config = config

        logger.info(f"Processed inline kind: Config '{config_name}'")
        return None
