"""Unified pipeline management system with base class and catalog functionality."""

import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Type

import yaml

from hexai.agent_factory.yaml_builder import YamlPipelineBuilder
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph


class PipelineDefinition(ABC):
    """Base class for pipeline definitions."""

    def __init__(self, yaml_path: str | None = None) -> None:
        """Initialize the pipeline definition.

        Args
        ----
            yaml_path: Optional path to YAML configuration file
        """
        self.builder = YamlPipelineBuilder()
        self._yaml_path = yaml_path
        self._config: dict[str, Any] | None = None
        self._register_functions()
        if yaml_path:
            self._load_yaml(yaml_path)

    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Pipeline description."""
        pass

    @abstractmethod
    def _register_functions(self) -> None:
        """Register all pipeline functions with the builder.

        This method should call self.builder.register_function() for each function.
        """
        pass

    def _load_yaml(self, yaml_path: str) -> None:
        """Load YAML configuration from the specified path."""
        self._yaml_path = yaml_path
        with open(yaml_path) as f:
            self._config = yaml.safe_load(f.read())

    async def execute(
        self,
        input_data: Any | None = None,
        ports: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the pipeline with given input and ports.

        Args
        ----
            input_data: Input data for the pipeline (default: {})
            ports: Port implementations (default: {})

        Returns
        -------
            Execution results dictionary with 'status', 'results', and 'trace'
        """
        if input_data is None:
            input_data = {}
        if ports is None:
            ports = {}

        try:
            if not self._yaml_path:
                raise ValueError(f"No pipeline YAML found for {self.name}")

            # Build and execute
            graph, pipeline_metadata = self.builder.build_from_yaml_file(self._yaml_path)

            # Create orchestrator with pipeline-specific field mapping
            field_mapping_mode = pipeline_metadata.get("field_mapping_mode", "default")
            custom_field_mappings = pipeline_metadata.get("custom_field_mappings")

            orchestrator = Orchestrator(field_mapping_mode=field_mapping_mode)
            results = await orchestrator.run(
                graph,
                input_data,
                additional_ports=ports,
                custom_field_mappings=custom_field_mappings,
            )

            return {"status": "success", "results": results}
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc().splitlines(),
            }

    def get_config(self) -> dict[str, Any] | None:
        """Get the loaded pipeline configuration."""
        return self._config

    def validate(self) -> dict[str, Any]:
        """Validate the pipeline configuration.

        Returns
        -------
            Validation result with 'valid' and 'errors' keys
        """
        if not self._yaml_path:
            return {"valid": False, "errors": ["No pipeline YAML found"]}

        try:
            graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
            return {"valid": True, "errors": []}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def get_input_type(self) -> Any:
        """Get the input data type from the first node(s) in the pipeline.

        Returns
        -------
            Input type from the first node(s)
        """
        if not self._yaml_path:
            return None

        # First try YAML-based schema extraction (most reliable)
        yaml_primitives = self._extract_input_schema_from_yaml()
        if yaml_primitives:
            return yaml_primitives

        # Fallback to node introspection
        try:
            graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
            waves = graph.waves()

            if not waves or not waves[0]:
                return None

            # Get the first wave of nodes
            first_wave = waves[0]

            # If multiple first nodes, return dictionary
            if len(first_wave) > 1:
                input_types = {}
                for node_name in first_wave:
                    node = graph.nodes[node_name]
                    # First try to get in_type attribute
                    if hasattr(node, "in_type") and node.in_type is not None:
                        input_types[node_name] = node.in_type
                    # Fallback to params.input_schema
                    elif hasattr(node, "params") and node.params:
                        input_schema = node.params.get("input_schema")
                        if input_schema:
                            input_types[node_name] = input_schema
                return input_types if input_types else None

            # Single first node - return its type directly
            first_node_name = first_wave[0]
            first_node = graph.nodes[first_node_name]

            # First try to get in_type attribute
            if hasattr(first_node, "in_type") and first_node.in_type is not None:
                return first_node.in_type
            # Fallback to params.input_schema
            elif hasattr(first_node, "params") and first_node.params:
                return first_node.params.get("input_schema")

            return None
        except Exception:
            return None

    def get_output_type(self) -> Any:
        """Get the output data type from the last node(s) in the pipeline.

        Returns
        -------
            Output type from the last node(s)
        """
        if not self._yaml_path:
            return None

        try:
            graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
            waves = graph.waves()

            if not waves:
                return None

            # Get the last wave of nodes
            last_wave = waves[-1]

            # If multiple last nodes, return dictionary
            if len(last_wave) > 1:
                output_types = {}
                for node_name in last_wave:
                    node = graph.nodes[node_name]
                    # First try to get out_type attribute
                    if hasattr(node, "out_type") and node.out_type:
                        output_types[node_name] = node.out_type
                    # Fallback to params.output_schema
                    elif hasattr(node, "params") and node.params:
                        output_schema = node.params.get("output_schema")
                        if output_schema:
                            output_types[node_name] = output_schema
                return output_types if output_types else None

            # Single last node - return its type directly
            last_node_name = last_wave[0]
            node = graph.nodes[last_node_name]

            # First try to get out_type attribute
            if hasattr(node, "out_type") and node.out_type:
                return node.out_type
            # Fallback to params.output_schema
            elif hasattr(node, "params") and node.params:
                return node.params.get("output_schema")

            return None
        except Exception:
            return None

    def get_node_types(self) -> dict[str, dict[str, Any]]:
        """Get type information for all nodes in the pipeline.

        Returns
        -------
            Dictionary mapping node names to their input/output type information
        """
        if not self._yaml_path:
            return {}

        try:
            graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
            node_types = {}

            for node_name, node_spec in graph.nodes.items():
                node_types[node_name] = {
                    "name": node_name,
                    "input_type": node_spec.in_type,
                    "output_type": node_spec.out_type,
                }

                # Add function name if available
                if hasattr(node_spec.fn, "__name__"):
                    node_types[node_name]["function"] = node_spec.fn.__name__

            return node_types
        except Exception:
            return {}

    def build_graph(self) -> DirectedGraph:
        """Build and return the pipeline's DirectedGraph for visualization."""
        if not self._yaml_path:
            raise ValueError(f"No YAML file found for pipeline {self.name}")
        graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
        return graph

    def get_graph(self) -> "DirectedGraph":
        """Get the built DirectedGraph for this pipeline.

        Returns
        -------
            DirectedGraph instance
        """
        if not self._yaml_path:
            raise ValueError(f"No pipeline YAML found for {self.name}")

        graph, metadata = self.builder.build_from_yaml_file(self._yaml_path)
        return graph

    def get_input_primitives(self) -> dict[str, str]:
        """Get the primitive input parameters that users need to provide to run this pipeline.

        Returns
        -------
            Dictionary mapping parameter names to their types as strings
        """
        try:
            # First try to get schema directly from YAML file
            yaml_primitives = self._extract_input_schema_from_yaml()
            if yaml_primitives:
                return yaml_primitives

            # Fallback to original method using first node's input type
            input_type = self.get_input_type()

            if input_type is None:
                return {}

            # Handle Pydantic models
            if hasattr(input_type, "model_fields"):
                primitives = {}
                for field_name, field_info in input_type.model_fields.items():
                    # Get the field type
                    field_type = field_info.annotation
                    if hasattr(field_type, "__name__"):
                        type_name = field_type.__name__
                    else:
                        type_name = str(field_type)
                    primitives[field_name] = type_name
                return primitives

            # Handle basic types
            elif hasattr(input_type, "__name__"):
                return {"input": input_type.__name__}

            # Handle dict types with annotations
            elif isinstance(input_type, dict):
                return input_type

            # Fallback
            else:
                return {"input": str(input_type)}

        except Exception:
            return {}

    def _extract_input_schema_from_yaml(self) -> dict[str, str]:
        """Extract input schema directly from YAML file with support for comments and defaults.

        Returns
        -------
            Dictionary mapping parameter names to their types with optional indicators
        """
        if not self._yaml_path or not os.path.exists(self._yaml_path):
            return {}

        try:
            with open(self._yaml_path, encoding="utf-8") as f:
                content = f.read()

            # Parse YAML
            config = yaml.safe_load(content)

            # Get input_schema from pipeline level (preferred)
            input_schema = config.get("input_schema", {})

            # Fallback to first node's input_schema (legacy support)
            if not input_schema:
                nodes = config.get("nodes", [])
                # Find first node (entry point)
                first_node = None
                for node in nodes:
                    if not node.get("depends_on"):  # Empty or no depends_on
                        first_node = node
                        break

                if first_node:
                    input_schema = first_node.get("params", {}).get("input_schema", {})

            if not input_schema:
                return {}

                # Parse schema with support for defaults and comments
            parsed_schema = {}
            for field_name, field_spec in input_schema.items():
                if isinstance(field_spec, str):
                    # Parse enhanced format: "str  # optional, default: value"
                    type_spec = field_spec.strip()

                    # Check for comment
                    comment_part = ""
                    if "#" in type_spec:
                        type_spec, comment_part = type_spec.split("#", 1)
                        type_spec = type_spec.strip()
                        comment_part = comment_part.strip()

                    # Check if field is marked as optional in comment
                    is_optional = "optional" in comment_part.lower()

                    # Clean up the type name (remove any leftover assignment syntax)
                    type_name = type_spec.split("=")[0].strip()

                    # Format the display string
                    if is_optional:
                        parsed_schema[field_name] = f"{type_name} (optional)"
                    else:
                        parsed_schema[field_name] = type_name
                else:
                    # Handle other formats (dict, etc.)
                    parsed_schema[field_name] = str(field_spec)

            return parsed_schema

        except Exception:
            return {}


class PipelineCatalog:
    """Catalog for discovering and managing pipeline definitions."""

    def __init__(self) -> None:
        """Initialize the pipeline catalog."""
        self._pipelines: dict[str, Type[PipelineDefinition]] = {}

    def register_pipeline(self, pipeline_class: Type[PipelineDefinition]) -> None:
        """Manually register a pipeline class."""
        instance = pipeline_class()
        self._pipelines[instance.name] = pipeline_class

    def list_pipelines(self) -> list[dict[str, str]]:
        """List all available pipelines."""
        return [
            {
                "name": name,
                "description": cls().description,
                "module": cls.__module__,
            }
            for name, cls in self._pipelines.items()
        ]

    def get_pipeline(self, name: str) -> PipelineDefinition | None:
        """Get a pipeline instance by name."""
        pipeline_class = self._pipelines.get(name)
        return pipeline_class() if pipeline_class else None

    async def execute_pipeline(
        self,
        name: str,
        input_data: Any | None = None,
        ports: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a pipeline by name."""
        pipeline = self.get_pipeline(name)
        if pipeline:
            return await pipeline.execute(input_data, ports)
        else:
            return {"status": "error", "error": f"Pipeline '{name}' not found"}


def get_catalog() -> PipelineCatalog:
    """Get the global pipeline catalog instance."""
    return PipelineCatalog()
