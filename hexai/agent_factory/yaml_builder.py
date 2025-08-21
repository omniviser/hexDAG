"""Pipeline Builder - Converts YAML configurations to DirectedGraphs.

Simple pipeline builder that focuses on basic YAML processing with simple data mapping.
"""

import logging
from typing import Any

from hexai.core.application.events.manager import PipelineEventManager
from hexai.core.application.nodes import NodeFactory
from hexai.core.application.prompt.template import ChatPromptTemplate
from hexai.core.domain.dag import DirectedGraph

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
        from hexai.utils.imports import optional_import

        yaml = optional_import("yaml", "cli")
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(f"Invalid YAML content: {e}") from e

        graph = DirectedGraph()

        # Extract pipeline-wide metadata
        pipeline_metadata = self._extract_pipeline_metadata(config)

        # Build nodes with simple processing and data mapping
        for node_config in config["nodes"]:
            node_id = node_config["id"]
            node_type = node_config.get("type", "function")
            params = node_config.get("params", {})
            deps = node_config.get("depends_on", [])

            # Handle input_mapping for data flow
            input_mapping = params.get("input_mapping")

            if input_mapping:
                # Validate input_mapping format
                if not isinstance(input_mapping, dict):
                    raise YamlPipelineBuilderError(
                        f"input_mapping for node '{node_id}' must be a dictionary"
                    )
                # Store input_mapping in params for the orchestrator to use
                params["input_mapping"] = input_mapping
                logger.debug(f"Node '{node_id}' has input mapping: {input_mapping}")

            # Auto-convert LLM nodes with incompatible template + schema combinations
            if node_type == "llm":
                logger.debug(f"📋 LLM node '{node_id}' original params: {list(params.keys())}")
                params = self._auto_convert_llm_node(node_id, params)
                logger.debug(f"📋 LLM node '{node_id}' final params: {list(params.keys())}")

            # Resolve function references (no schema inference)
            if node_type == "function" and "fn" in params:
                func_ref = params["fn"]
                if isinstance(func_ref, str) and func_ref in self.registered_functions:
                    actual_func = self.registered_functions[func_ref]
                    params["fn"] = actual_func

            # Create node using NodeFactory (let nodes handle their own logic)
            node = NodeFactory.create_node(node_type, node_id, **params)

            # Add dependencies
            if deps:
                if isinstance(deps, list):
                    node = node.after(*deps)
                else:
                    node = node.after(deps)

            graph.add(node)

        logger.info(f"✅ Built pipeline with {len(graph.nodes)} nodes")
        logger.info(
            f"🔧 Field mapping mode: {pipeline_metadata.get('field_mapping_mode', 'default')}"
        )

        # Validate data mapping after building
        warnings = self.validate_data_mapping(config)
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️  Data mapping: {warning}")

        return graph, pipeline_metadata

    def _extract_pipeline_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract pipeline-wide metadata from YAML configuration.

        Returns
        -------
            dict: Pipeline metadata including field mapping configuration
        """
        metadata = {
            "name": config.get("name"),
            "description": config.get("description"),
            "field_mapping_mode": config.get("field_mapping_mode", "default"),
        }

        # Validate field mapping mode
        field_mapping_mode = metadata["field_mapping_mode"]
        if field_mapping_mode not in ["none", "default", "custom"]:
            raise YamlPipelineBuilderError(
                f"Invalid field_mapping_mode '{field_mapping_mode}'. "
                f"Must be 'none', 'default', or 'custom'"
            )

        # Handle custom field mappings
        custom_field_mappings = config.get("custom_field_mappings")
        if field_mapping_mode == "custom":
            if not custom_field_mappings:
                raise YamlPipelineBuilderError(
                    "custom_field_mappings required when field_mapping_mode='custom'"
                )
            metadata["custom_field_mappings"] = custom_field_mappings
        elif custom_field_mappings:
            # Store custom mappings even if not in custom mode (for potential future use)
            metadata["custom_field_mappings"] = custom_field_mappings

        # Extract other pipeline-wide configurations
        for key in ["version", "author", "tags", "environment"]:
            if key in config:
                metadata[key] = config[key]

        return metadata

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
        """Validate data mapping in YAML configuration.

        Returns a list of validation warnings (not errors).
        """
        warnings = []
        nodes = config.get("nodes", [])
        node_names = {node.get("id") for node in nodes}

        for node_config in nodes:
            node_id = node_config.get("id")
            params = node_config.get("params", {})
            input_mapping = params.get("input_mapping")
            dependencies = set(node_config.get("depends_on", []))

            # Validate input_mapping
            if input_mapping:
                warnings.extend(
                    self._validate_mapping_references(
                        node_id, input_mapping, node_names, dependencies, "input_mapping"
                    )
                )

        return warnings

    def _validate_mapping_references(
        self,
        node_id: str,
        mapping: dict[str, Any],
        node_names: set[str],
        dependencies: set[str],
        mapping_type: str,
    ) -> list[str]:
        """Validate mapping references and return warnings."""
        warnings = []

        for _target_field, source_path in mapping.items():
            # Skip validation for non-string values (like defaults)
            if not isinstance(source_path, str):
                continue

            if "." in source_path:
                node_name, _field_name = source_path.split(".", 1)
                if node_name not in node_names:
                    warnings.append(
                        f"Node '{node_id}' {mapping_type} references unknown node '{node_name}'"
                    )
                elif node_name not in dependencies:
                    warnings.append(
                        f"Node '{node_id}' {mapping_type} references '{node_name}' "
                        f"but it's not in dependencies"
                    )
            else:
                # Direct node reference
                if source_path not in node_names:
                    warnings.append(
                        f"Node '{node_id}' {mapping_type} references unknown node '{source_path}'"
                    )

        return warnings
