"""Schema extraction and processing for DAG visualization."""

import inspect
import logging
from pathlib import Path
from typing import Any, get_type_hints

from hexai.core.domain.dag import DirectedGraph

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Extract and format schemas from nodes for visualization."""

    def extract_all_schemas(self, graph: DirectedGraph) -> dict[str, dict[str, Any]]:
        """Extract schemas for all nodes in the graph.

        Args:
            graph: The DirectedGraph to extract schemas from

        Returns:
            Dictionary mapping node names to their schema information
        """
        schemas = {}

        # Try to load compiled schemas if pipeline name is available
        pipeline_name = getattr(graph, "_pipeline_name", None)
        if pipeline_name:
            compiled_schemas, _ = self.load_compiled_schemas(pipeline_name)
            if compiled_schemas:
                return compiled_schemas

        # Fallback to extracting schemas from node specs
        for node_name, node_spec in graph.nodes.items():
            input_schema, output_schema = self.extract_node_schemas(node_spec)
            if input_schema or output_schema:
                schemas[node_name] = {
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                    "type": getattr(node_spec, "type", None),
                }

        return schemas

    def extract_node_schemas(
        self, node_spec: Any
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """Extract input and output schemas from a node specification.

        Args:
            node_spec: Node specification object

        Returns:
            Tuple of (input_schema, output_schema)
        """
        input_schema = self._extract_node_input_schema(node_spec)
        output_schema = self._extract_node_output_schema(node_spec)
        return input_schema, output_schema

    def _extract_node_input_schema(self, node_spec: Any) -> dict[str, str] | None:
        """Extract input schema information from a node specification.

        Args:
            node_spec: Node specification object

        Returns:
            Dictionary of input schema fields or None
        """
        # Check if node has input type information
        if hasattr(node_spec, "in_model") and node_spec.in_model:
            return self._convert_type_to_schema_dict(node_spec.in_model)

        # Check for function-specific schema info
        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__annotations__"):
            return self.extract_function_input_schema(node_spec.fn)

        return None

    def _extract_node_output_schema(self, node_spec: Any) -> dict[str, str] | None:
        """Extract output schema information from a node specification.

        Args:
            node_spec: Node specification object

        Returns:
            Dictionary of output schema fields or None
        """
        # Check if node has output type information
        if hasattr(node_spec, "out_model") and node_spec.out_model:
            return self._convert_type_to_schema_dict(node_spec.out_model)

        # Check for function-specific schema info
        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__annotations__"):
            return self.extract_function_output_schema(node_spec.fn)

        return None

    def extract_function_schemas(
        self, func: Any
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """Extract input and output schemas from function type hints.

        Args:
            func: Function to analyze

        Returns:
            Tuple of (input_schema, output_schema)
        """
        input_schema = self.extract_function_input_schema(func)
        output_schema = self.extract_function_output_schema(func)
        return input_schema, output_schema

    def extract_function_input_schema(self, func: Any) -> dict[str, str] | None:
        """Extract input schema from function type hints.

        Args:
            func: Function to analyze

        Returns:
            Dictionary of input schema fields or None
        """
        try:
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            if params and params[0].name != "self":
                first_param = params[0]
                param_type = hints.get(first_param.name)
                return self._convert_type_to_schema_dict(param_type)

        except Exception:
            # Function signature analysis failed
            pass  # nosec B110
        return None

    def extract_function_output_schema(self, func: Any) -> dict[str, str] | None:
        """Extract output schema from function return type hints.

        Args:
            func: Function to analyze

        Returns:
            Dictionary of output schema fields or None
        """
        try:
            hints = get_type_hints(func)
            return_type = hints.get("return")

            if return_type and return_type is not type(None):
                return self._convert_type_to_schema_dict(return_type)

        except Exception:
            # Return type analysis failed
            pass  # nosec B110
        return None

    def load_compiled_schemas(
        self, pipeline_name: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None]:
        """Load schema information by compiling the pipeline on-the-fly.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Tuple of (node_schemas_dict, pipeline_input_schema)
        """
        try:
            # Try to import the compiler
            try:
                from hexai.agent_factory.compiler import compile_pipeline
            except ImportError:
                logger.debug("Pipeline compiler not available, skipping schema enhancement")
                return {}, None

            # Find the pipeline YAML file
            current_dir = Path.cwd()
            possible_paths = [
                current_dir
                / f"src/pipelines/{pipeline_name.replace('_pipeline', '')}/pipeline.yaml",
                current_dir / f"src/pipelines/{pipeline_name}/pipeline.yaml",
                current_dir / f"{pipeline_name}.yaml",
            ]

            yaml_file = None
            for path in possible_paths:
                if path.exists():
                    yaml_file = path
                    break

            if not yaml_file:
                logger.debug("Pipeline YAML not found for %s", pipeline_name)
                return {}, None

            # Compile on-the-fly to get all schemas
            try:
                compiled_data = compile_pipeline(yaml_file)
            except Exception as e:
                logger.debug("Failed to compile pipeline %s: %s", pipeline_name, e)
                return {}, None

            # Extract node schemas from compiled data
            node_schemas = {}
            if compiled_data:
                for node_config in compiled_data.node_configs:
                    node_id = node_config["id"]
                    params = node_config.get("params", {})
                    node_schemas[node_id] = {
                        "input_schema": params.get("input_schema"),
                        "output_schema": params.get("output_schema"),
                        "type": node_config.get("type", "unknown"),
                        "function_name": (
                            params.get("fn") if node_config.get("type") == "function" else None
                        ),
                    }

            return node_schemas, compiled_data.input_schema if compiled_data else None

        except Exception as e:
            logger.debug("Exception in schema loading: %s", e)
            return {}, None

    def format_schema_label(self, label: str, schema: Any) -> str:
        """Format a schema for display in a node label.

        Args:
            label: Base label
            schema: Schema information

        Returns:
            Formatted label string
        """
        if schema is None:
            return label

        # Handle Pydantic models
        if hasattr(schema, "__name__") and hasattr(schema, "model_fields"):
            model_fields = schema.model_fields
            field_lines = []

            for field_name, field_info in model_fields.items():
                field_type = getattr(field_info.annotation, "__name__", str(field_info.annotation))

                if hasattr(field_info, "default") and field_info.default is not ...:
                    if field_info.default is None:
                        field_line = f"{field_name}: {field_type} = None"
                    else:
                        field_line = f"{field_name}: {field_type} = {field_info.default}"
                else:
                    field_line = f"{field_name}: {field_type}"

                field_lines.append(field_line)

            if len(field_lines) <= 3:
                field_str = "\\n".join(field_lines)
            else:
                field_str = "\\n".join(field_lines[:3]) + "\\n..."

            return f"{label}\\n{schema.__name__}\\n{field_str}"

        # Handle dict schemas
        if isinstance(schema, dict):
            field_lines = []
            for key, value in schema.items():
                if isinstance(value, str):
                    field_lines.append(f"{key}: {value}")
                else:
                    field_lines.append(key)

            if len(field_lines) <= 4:
                field_str = "\\n".join(field_lines)
            else:
                field_str = "\\n".join(field_lines[:4]) + "\\n..."
            return f"{label}\\n{field_str}"

        # Handle type annotations
        if isinstance(schema, type):
            return f"{label}\\n({schema.__name__})"

        # String representation
        schema_str = str(schema)
        if len(schema_str) > 30:
            schema_str = schema_str[:27] + "..."
        return f"{label}\\n({schema_str})"

    def _convert_type_to_schema_dict(self, type_obj: Any) -> dict[str, str] | None:
        """Convert a type object to a schema dictionary.

        Args:
            type_obj: Type object to convert

        Returns:
            Dictionary representation of the type
        """
        try:
            # Handle Pydantic models
            if hasattr(type_obj, "model_fields"):
                schema = {}
                for field_name, field_info in type_obj.model_fields.items():
                    field_type = getattr(
                        field_info.annotation, "__name__", str(field_info.annotation)
                    )
                    schema[field_name] = field_type
                return schema

            if hasattr(type_obj, "__annotations__"):
                schema = {}
                for field_name, field_type in type_obj.__annotations__.items():
                    type_name = getattr(field_type, "__name__", str(field_type))
                    schema[field_name] = type_name
                return schema

            # Handle dict types
            if isinstance(type_obj, dict):
                return type_obj

        except Exception:
            # Type conversion failed
            pass  # nosec B110
        return None
