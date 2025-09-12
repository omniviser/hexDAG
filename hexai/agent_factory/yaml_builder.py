"""Pipeline Builder - Converts YAML configurations to DirectedGraphs.

Simple pipeline builder that focuses on basic YAML processing with simple data mapping.
"""

import logging
from typing import Any

import yaml

from hexai.agent_factory.yaml_validator import YamlValidator
from hexai.core.application.events.manager import PipelineEventManager
from hexai.core.application.nodes.mapped_input import FieldMappingRegistry
from hexai.core.application.prompt.template import ChatPromptTemplate
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.domain.dag import DirectedGraph
from hexai.core.registry import registry

logger = logging.getLogger(__name__)


class YamlPipelineBuilderError(Exception):
    """Custom exception for YAML pipeline building errors."""

    pass


class YamlPipelineBuilder:
    """Simple pipeline builder with basic data mapping support and intelligent auto-conversion."""

    def __init__(self, event_manager: PipelineEventManager | None = None) -> None:
        """Initialize the pipeline builder.

        Args
        ----
            event_manager: Optional event manager for observer pattern
        """
        self.registered_functions: dict[str, Any] = {}
        self.event_manager = event_manager or PipelineEventManager()
        self.field_mapping_registry = FieldMappingRegistry()
        self.validator = YamlValidator()

    def register_function(self, name: str, func: Any) -> None:
        """Register a function for use in YAML pipelines."""
        self.registered_functions[name] = func

    def build_from_yaml_file(self, yaml_file_path: str) -> tuple[DirectedGraph, dict[str, Any]]:
        """Build DirectedGraph from YAML file."""
        try:
            with open(yaml_file_path, encoding="utf-8") as file:
                yaml_content = file.read()
            return self.build_from_yaml_string(yaml_content)
        except OSError as e:
            raise YamlPipelineBuilderError(
                f"Failed to load YAML file '{yaml_file_path}': {e}"
            ) from e

    def build_from_yaml_string(self, yaml_content: str) -> tuple[DirectedGraph, dict[str, Any]]:
        """Convert YAML string to DirectedGraph.

        Simple approach with basic data mapping support and intelligent auto-conversion.
        Supports input_mapping for explicit field mapping between nodes.
        Automatically converts incompatible LLM configurations to compatible formats.

        Returns
        -------
            tuple: (DirectedGraph, pipeline_metadata)
        """
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(f"Invalid YAML content: {e}") from e

        # Validate configuration
        validation_result = self.validator.validate(config)

        if not validation_result.is_valid:
            error_msg = "YAML validation failed:\n"
            for error in validation_result.errors:
                error_msg += f"  ERROR: {error}\n"
            raise YamlPipelineBuilderError(error_msg)

        # Log warnings and suggestions
        for warning in validation_result.warnings:
            logger.warning("YAML validation warning: %s", warning)
        for suggestion in validation_result.suggestions:
            logger.info("YAML validation suggestion: %s", suggestion)

        graph = DirectedGraph()

        # Extract pipeline-wide metadata and register common mappings
        pipeline_metadata = self._extract_pipeline_metadata(config)
        self._register_common_mappings(config)

        # Build nodes with simple processing and data mapping
        for node_config in config["nodes"]:
            node_id = node_config["id"]
            node_type = node_config.get("type", "function")
            params = node_config.get("params", {})
            deps = node_config.get("depends_on", [])

            # Handle field_mapping parameter
            field_mapping = params.get("field_mapping")
            if field_mapping:
                # Resolve mapping (could be a string reference or inline dict)
                resolved_mapping = self.field_mapping_registry.get(field_mapping)
                params["field_mapping"] = resolved_mapping
                logger.debug("Node '%s' using field mapping: %s", node_id, resolved_mapping)

            # Auto-convert LLM nodes with incompatible template + schema combinations
            if node_type == "llm":
                logger.debug("📋 LLM node '%s' original params: %s", node_id, list(params.keys()))
                params = self._auto_convert_llm_node(node_id, params)
                logger.debug("📋 LLM node '%s' final params: %s", node_id, list(params.keys()))

            # Resolve function references (no schema inference)
            if node_type == "function" and "fn" in params:
                func_ref = params["fn"]
                if isinstance(func_ref, str) and func_ref in self.registered_functions:
                    actual_func = self.registered_functions[func_ref]
                    params["fn"] = actual_func

            # Create node using NodeFactory (let nodes handle their own logic)
            # Ensure registry is bootstrapped
            ensure_bootstrapped()

            # Get node factory from registry
            factory_name = f"{node_type}_node"
            factory = registry.get(factory_name, namespace="core")

            # Ensure factory is callable
            if not callable(factory):
                raise TypeError(
                    f"Expected callable factory for {factory_name}, got {type(factory)}"
                )

            # Create node using factory
            node = factory(node_id, **params)

            # Add dependencies
            if deps:
                if isinstance(deps, list):
                    node = node.after(*deps)
                else:
                    node = node.after(deps)

            graph.add(node)

        logger.info("✅ Built pipeline with %d nodes", len(graph.nodes))

        return graph, pipeline_metadata

    def _extract_pipeline_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract pipeline-wide metadata from YAML configuration.

        Returns
        -------
            dict: Pipeline metadata
        """
        metadata = {
            "name": config.get("name"),
            "description": config.get("description"),
        }

        # Include common_field_mappings in metadata
        if "common_field_mappings" in config:
            metadata["common_field_mappings"] = config["common_field_mappings"]

        # Extract other pipeline-wide configurations
        for key in ["version", "author", "tags", "environment"]:
            if key in config:
                metadata[key] = config[key]

        return metadata

    def _register_common_mappings(self, config: dict[str, Any]) -> None:
        """Register common field mappings from config.

        Args
        ----
            config: Pipeline configuration
        """
        common_mappings = config.get("common_field_mappings", {})
        for name, mapping in common_mappings.items():
            self.field_mapping_registry.register(name, mapping)
            logger.debug("Registered common field mapping '%s': %s", name, mapping)

    def _auto_convert_llm_node(self, node_id: str, params: dict[str, Any]) -> dict[str, Any]:
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

    def validate_data_mapping(self, config: dict[str, Any]) -> list[str]:
        """Validate field mappings in YAML configuration.

        Returns a list of validation warnings (not errors).
        """
        warnings = []

        # Validate common_field_mappings
        common_mappings = config.get("common_field_mappings", {})
        if common_mappings and not isinstance(common_mappings, dict):
            warnings.append("common_field_mappings must be a dictionary")

        # Validate node field_mapping references
        nodes = config.get("nodes", [])
        for node_config in nodes:
            node_id = node_config.get("id")
            params = node_config.get("params", {})
            field_mapping = params.get("field_mapping")

            if field_mapping:
                # If it's a string, check it references a known common mapping
                if isinstance(field_mapping, str):
                    if field_mapping not in common_mappings:
                        warnings.append(
                            f"Node '{node_id}' references unknown field mapping '{field_mapping}'"
                        )
                # If it's a dict, validate it's properly formatted
                elif isinstance(field_mapping, dict):
                    for target, source in field_mapping.items():
                        if not isinstance(source, str):
                            warnings.append(
                                f"Node '{node_id}' field mapping has invalid source for '{target}'"
                            )
                else:
                    warnings.append(
                        f"Node '{node_id}' field_mapping must be a string reference or dict"
                    )

        return warnings
