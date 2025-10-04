"""Pipeline Builder - Converts YAML configurations to DirectedGraphs.

Simple pipeline builder that focuses on basic YAML processing with simple data mapping.
"""

from pathlib import Path
from typing import Any

import yaml

from hexai.agent_factory.yaml_validator import YamlValidator

# ObserverManager is now a port - passed as dependency
from hexai.core.application.nodes.mapped_input import FieldMappingRegistry
from hexai.core.application.prompt.template import ChatPromptTemplate
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.logging import get_logger
from hexai.core.registry import registry
from hexai.core.registry.models import NAMESPACE_SEPARATOR

logger = get_logger(__name__)


class YamlPipelineBuilderError(Exception):
    """Custom exception for YAML pipeline building errors."""

    pass


class YamlPipelineBuilder:
    """Simple pipeline builder with basic data mapping support and intelligent auto-conversion."""

    def __init__(self, event_manager: Any = None) -> None:
        """Initialize the pipeline builder.

        Args
        ----
            event_manager: Optional event manager for observer pattern
        """
        self.registered_functions: dict[str, Any] = {}

        self.event_manager = event_manager  # Now expects a port implementation
        self.field_mapping_registry = FieldMappingRegistry()
        self.validator = YamlValidator()

    def register_function(self, name: str, func: Any) -> None:
        """Register a function for use in YAML pipelines."""
        self.registered_functions[name] = func

    def build_from_yaml_file(self, yaml_file_path: str) -> tuple[DirectedGraph, dict[str, Any]]:
        """Build DirectedGraph from YAML file.

        Parameters
        ----------
        yaml_file_path : str
            Path to the YAML file to build from

        Returns
        -------
        tuple[DirectedGraph, dict[str, Any]]
            Tuple of (DirectedGraph, pipeline_metadata)

        Raises
        ------
        YamlPipelineBuilderError
            If the YAML file cannot be loaded or parsed
        """
        try:
            yaml_path = Path(yaml_file_path)
            with yaml_path.open(encoding="utf-8") as file:
                yaml_content = file.read()
            return self.build_from_yaml_string(yaml_content)
        except OSError as e:
            raise YamlPipelineBuilderError(
                f"Failed to load YAML file '{yaml_file_path}': {e}"
            ) from e

    def build_from_yaml_string(self, yaml_content: str) -> tuple[DirectedGraph, dict[str, Any]]:
        """Convert declarative YAML manifest to DirectedGraph.

        Expects declarative manifest format with K8s-style structure:
        ```yaml
        apiVersion: v1
        kind: Pipeline
        metadata:
          name: my-pipeline
        spec:
          nodes:
            - kind: llm_node
              metadata:
                name: processor
              spec:
                prompt_template: "Process {{input}}"
                dependencies: []
        ```

        Parameters
        ----------
        yaml_content : str
            YAML content string to parse

        Returns
        -------
        tuple[DirectedGraph, dict[str, Any]]
            Tuple of (DirectedGraph, pipeline_metadata)
        """
        config = self._parse_and_validate_yaml(yaml_content)
        pipeline_metadata = self._extract_pipeline_metadata(config)
        self._register_common_mappings(config)

        graph = self._build_graph_from_config(config)

        logger.info("✅ Built pipeline with {count} nodes", count=len(graph.nodes))
        return graph, pipeline_metadata

    def _parse_and_validate_yaml(self, yaml_content: str) -> dict[str, Any]:
        """Parse YAML content and validate the configuration.

        Parameters
        ----------
        yaml_content : str
            YAML content string to parse

        Returns
        -------
        dict[str, Any]
            Validated configuration dictionary

        Raises
        ------
        YamlPipelineBuilderError
            If YAML parsing or validation fails
        """
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(f"Invalid YAML content: {e}") from e

        # Ensure config is a dictionary
        if not isinstance(config, dict):
            raise YamlPipelineBuilderError(
                f"YAML root must be a dictionary, got {type(config).__name__}"
            )

        # Validate manifest format
        self._validate_manifest_format(config)

        # Validate configuration
        validation_result = self.validator.validate(config)

        if not validation_result.is_valid:
            error_msg = "YAML validation failed:\n"
            for error in validation_result.errors:
                error_msg += f"  ERROR: {error}\n"
            raise YamlPipelineBuilderError(error_msg)

        # Log warnings and suggestions
        for warning in validation_result.warnings:
            logger.warning("YAML validation warning: {msg}", msg=warning)
        for suggestion in validation_result.suggestions:
            logger.info("YAML validation suggestion: {msg}", msg=suggestion)

        return config

    def _build_graph_from_config(self, config: dict[str, Any]) -> DirectedGraph:
        """Build DirectedGraph from validated configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Validated YAML configuration

        Returns
        -------
        DirectedGraph
            Constructed graph with all nodes
        """
        graph = DirectedGraph()
        nodes_list = config.get("spec", {}).get("nodes", [])

        for node_config in nodes_list:
            node = self._build_node_from_config(node_config)
            graph.add(node)

        return graph

    def _build_node_from_config(self, node_config: dict[str, Any]) -> NodeSpec:
        """Build a single NodeSpec from node configuration.

        Parameters
        ----------
        node_config : dict[str, Any]
            Node configuration from YAML

        Returns
        -------
        NodeSpec
            Constructed node specification
        """
        # Extract node configuration
        node_id = node_config.get("metadata", {}).get("name")
        node_type, namespace = self._parse_kind(node_config["kind"])
        params = node_config.get("spec", {}).copy()  # Copy to avoid modifying original
        deps = params.pop("dependencies", [])

        # Process parameters
        params = self._process_node_params(node_id, node_type, params)

        # Create node using factory
        node = self._create_node_from_factory(node_id, node_type, namespace, params)

        # Add dependencies
        if deps:
            node = node.after(*deps) if isinstance(deps, list) else node.after(deps)

        return node

    def _process_node_params(
        self, node_id: str, node_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Process and transform node parameters.

        Parameters
        ----------
        node_id : str
            Node identifier
        node_type : str
            Type of node (e.g., "llm", "function")
        params : dict[str, Any]
            Raw node parameters

        Returns
        -------
        dict[str, Any]
            Processed parameters
        """
        # Handle field_mapping parameter
        field_mapping = params.get("field_mapping")
        if field_mapping:
            resolved_mapping = self.field_mapping_registry.get(field_mapping)
            params["field_mapping"] = resolved_mapping
            logger.debug(
                "Node '{node_id}' using field mapping: {mapping}",
                node_id=node_id,
                mapping=resolved_mapping,
            )

        # Auto-convert LLM nodes with incompatible template + schema combinations
        if node_type == "llm":
            logger.debug(
                "📋 LLM node '{node_id}' original params: {params}",
                node_id=node_id,
                params=list(params.keys()),
            )
            params = self._auto_convert_llm_node(node_id, params)
            logger.debug(
                "📋 LLM node '{node_id}' final params: {params}",
                node_id=node_id,
                params=list(params.keys()),
            )

        # Resolve function references
        if node_type == "function" and "fn" in params:
            func_ref = params["fn"]
            if isinstance(func_ref, str) and func_ref in self.registered_functions:
                params["fn"] = self.registered_functions[func_ref]

        return params

    @staticmethod
    def _create_node_from_factory(
        node_id: str, node_type: str, namespace: str, params: dict[str, Any]
    ) -> NodeSpec:
        """Create NodeSpec using registry factory.

        Parameters
        ----------
        node_id : str
            Node identifier
        node_type : str
            Type of node
        namespace : str
            Component namespace
        params : dict[str, Any]
            Processed node parameters

        Returns
        -------
        NodeSpec
            Created node specification

        Raises
        ------
        TypeError
            If factory is not callable or doesn't return NodeSpec
        """
        ensure_bootstrapped()

        factory_name = f"{node_type}_node"
        factory = registry.get(factory_name, namespace=namespace)

        if not callable(factory):
            raise TypeError(f"Expected callable factory for {factory_name}, got {type(factory)}")

        node = factory(node_id, **params)

        if not isinstance(node, NodeSpec):
            raise TypeError(f"Factory {factory_name} did not return a NodeSpec")

        return node

    @staticmethod
    def _validate_manifest_format(config: dict[str, Any]) -> None:
        """Validate that config uses declarative manifest format.

        Args
        ----
            config: Raw YAML configuration

        Raises
        ------
            YamlPipelineBuilderError: If config is not in manifest format
        """
        if "kind" not in config:
            raise YamlPipelineBuilderError(
                "YAML must use declarative manifest format with 'kind' field. "
                "Example:\n"
                "apiVersion: v1\n"
                "kind: Pipeline\n"
                "metadata:\n"
                "  name: my-pipeline\n"
                "spec:\n"
                "  nodes: [...]"
            )

        if "spec" not in config:
            raise YamlPipelineBuilderError("Manifest YAML must have 'spec' field")

        if "metadata" not in config:
            raise YamlPipelineBuilderError("Manifest YAML must have 'metadata' field")

    @staticmethod
    def _parse_kind(kind: str) -> tuple[str, str]:
        """Parse kind into (node_type, namespace).

        Supports both simple kinds and namespace-qualified kinds:
        - "llm_node" -> ("llm", "core")
        - "my-plugin:dalle_node" -> ("dalle", "my-plugin")

        Args
        ----
            kind: Kind string from YAML

        Returns
        -------
            Tuple of (node_type, namespace)
        """
        if NAMESPACE_SEPARATOR in kind:
            # Namespace-qualified kind
            namespace, node_kind = kind.split(NAMESPACE_SEPARATOR, 1)
        else:
            # Simple kind - assume core namespace
            namespace = "core"
            node_kind = kind

        # Remove '_node' suffix if present (e.g., "llm_node" -> "llm")
        node_type = node_kind.removesuffix("_node")

        return node_type, namespace

    @staticmethod
    def _extract_pipeline_metadata(config: dict[str, Any]) -> dict[str, Any]:
        """Extract pipeline-wide metadata from declarative YAML manifest.

        Returns
        -------
            dict: Pipeline metadata
        """
        metadata_section = config.get("metadata", {})
        spec = config.get("spec", {})

        metadata = {
            "name": metadata_section.get("name"),
            "description": metadata_section.get("description"),
        }

        # Include common_field_mappings from spec
        if "common_field_mappings" in spec:
            metadata["common_field_mappings"] = spec["common_field_mappings"]

        # Extract other metadata fields
        for key in ["version", "author", "tags", "environment"]:
            if key in metadata_section:
                metadata[key] = metadata_section[key]

        return metadata

    def _register_common_mappings(self, config: dict[str, Any]) -> None:
        """Register common field mappings from manifest config.

        Args
        ----
            config: Pipeline configuration
        """
        common_mappings = config.get("spec", {}).get("common_field_mappings", {})

        for name, mapping in common_mappings.items():
            self.field_mapping_registry.register(name, mapping)
            logger.debug(
                "Registered common field mapping '{name}': {mapping}", name=name, mapping=mapping
            )

    @staticmethod
    def _auto_convert_llm_node(node_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Auto-convert LLM node parameters to handle common configuration incompatibilities.

        Specifically handles the case where prompt_template (string) is used with output_schema,
        which requires structured templates. Automatically converts to ChatPromptTemplate.

        Args
        ----
            node_id: Node identifier for logging
            params: Original node parameters

        Returns
        -------
            Updated parameters with auto-conversions applied
        """
        # Create a copy to avoid modifying the original
        params = params.copy()

        # Check for incompatible prompt_template + output_schema combination
        has_string_template = "prompt_template" in params and isinstance(
            params["prompt_template"], str
        )
        has_output_schema = "output_schema" in params
        has_parse_as_json = params.get("parse_as_json", False)

        if has_string_template and (has_output_schema or has_parse_as_json):
            # Auto-convert string template to structured template
            original_template = params["prompt_template"]

            # Remove parse_as_json (incompatible with structured templates)
            if "parse_as_json" in params:
                del params["parse_as_json"]

            # Convert the string template to ChatPromptTemplate object
            # Keep the prompt_template parameter name but change the value type
            chat_template = ChatPromptTemplate(human_message=original_template)
            params["prompt_template"] = chat_template

            logger.info(
                f"🔄 Auto-converted LLM node '{node_id}': "
                f"prompt_template (string) + output_schema → prompt_template (ChatPromptTemplate)"
            )

        return params
