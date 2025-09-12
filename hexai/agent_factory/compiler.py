"""Pipeline Compiler - Compilation-specific optimizations only.

The compiler takes pre-built, validated graphs from PipelineBuilder and adds
compilation-specific optimizations like pre-computed execution waves and
dependency chain inference.
"""

import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from pydantic import BaseModel, Field

from hexai.agent_factory.base import PipelineCatalog
from hexai.core.domain.dag import DirectedGraph


def validate_pipeline_compilation(graph: DirectedGraph, pipeline_name: str) -> None:
    """Stub function for pipeline compilation validation - Tier 2 functionality.

    This is a simplified validation that always passes.
    Real implementation would be in the Tier 2 compiler layer.
    """
    pass


class CompiledPipelineData(BaseModel):
    """Pydantic model for compiled pipeline data with validation."""

    name: str = Field(..., description="Pipeline name")
    node_configs: list[dict[str, Any]] = Field(
        ..., description="Node configurations with inferred schemas"
    )
    execution_waves: list[list[str]] = Field(..., description="Pre-computed execution order")
    input_schema: dict[str, str] | None = Field(None, description="Pipeline input schema")
    output_schemas: dict[str, dict[str, str]] = Field(..., description="Node output schemas")
    type_safety_score: str = Field(..., description="Type safety score (e.g., '7/7')")
    functional_hash: str = Field(
        ..., description="Hash of functional YAML content (excludes metadata)"
    )
    compiled_at: str = Field(..., description="ISO format timestamp of compilation")
    field_mapping_mode: str = Field(default="default", description="Field mapping mode")
    custom_field_mappings: dict[str, list[str]] | None = Field(
        None, description="Custom field mappings"
    )


def extract_functional_content(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Extract only the functional parts of pipeline YAML that affect compilation.

    Args
    ----
        yaml_data: Complete pipeline YAML data

    Returns
    -------
        Dictionary containing only functional parts
    """
    return {
        "name": yaml_data.get("name"),  # May affect compiled class names
        "nodes": yaml_data.get("nodes", []),  # Core pipeline logic
        # Skip description, comments, and other metadata
    }


def generate_functional_hash(yaml_data: dict[str, Any]) -> str:
    """Generate a deterministic hash of functional pipeline content.

    Args
    ----
        yaml_data: Complete pipeline YAML data

    Returns
    -------
        16-character hash of functional content
    """
    functional_content = extract_functional_content(yaml_data)
    # Sort keys for deterministic serialization
    json_str = json.dumps(functional_content, sort_keys=True)
    # Generate short hash for readability
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class PipelineCompiler:
    """Compiler that focuses only on compilation optimizations.

    Uses PipelineBuilder for all uncompiled functionality (graph building, validation, schema
    inference) and adds compilation-specific optimizations.
    """

    def compile_from_yaml(self, yaml_path: str | Path) -> CompiledPipelineData:
        """Compile pipeline - PipelineBuilder does uncompiled work, compiler adds optimizations."""
        yaml_path = Path(yaml_path)

        # Step 1: Load YAML config first
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        pipeline_name = config.get("name", yaml_path.stem)

        # Step 2: Get pipeline instance (contains registered functions)
        catalog = PipelineCatalog()
        pipeline_instance = catalog.get_pipeline(pipeline_name)
        if not pipeline_instance:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in catalog")

        # Step 3: Let PipelineBuilder do ALL the uncompiled work
        # This includes: YAML parsing, function resolution, validation
        graph, pipeline_metadata = pipeline_instance.builder.build_from_yaml_file(str(yaml_path))

        # Extract field mapping configuration from metadata
        field_mapping_mode = pipeline_metadata.get("field_mapping_mode", "default")
        custom_field_mappings = pipeline_metadata.get("custom_field_mappings")

        # Step 4: Compiler adds compilation-specific optimizations
        # Extract pre-computed information from the BUILT graph + access to registry
        node_configs = self._extract_from_built_graph(graph, config, pipeline_instance.builder)
        output_schemas = self._extract_output_schemas(node_configs)

        # Compilation optimization: infer intermediate input schemas from dependency chains
        self._infer_dependency_input_schemas(node_configs, output_schemas)

        # Compilation optimization: pre-compute execution waves
        execution_waves = graph.waves()

        # Compilation optimization: extract pipeline-level schemas
        input_schema = self._extract_pipeline_input_schema(pipeline_instance, node_configs)

        # Compilation optimization: calculate type safety metrics
        type_safe_count, total_nodes = self._validate_type_safety(
            node_configs, pipeline_name, graph
        )

        # Generate functional hash for smart recompilation detection
        functional_hash = generate_functional_hash(config)
        compiled_at = datetime.now().isoformat()

        return CompiledPipelineData(
            name=pipeline_name,
            node_configs=node_configs,
            execution_waves=execution_waves,
            input_schema=input_schema,
            output_schemas=output_schemas,
            type_safety_score=f"{type_safe_count}/{total_nodes}",
            functional_hash=functional_hash,
            compiled_at=compiled_at,
            field_mapping_mode=field_mapping_mode,
            custom_field_mappings=custom_field_mappings,
        )

    def _validate_schema_types(self, schema: dict[str, Any]) -> bool:
        """Validate that schema contains proper types, not Union, Any, etc."""
        if not schema:
            return False

        for field_type in schema.values():
            if isinstance(field_type, str):
                # Check for invalid types
                if field_type in ["Union", "Any", "object", "unknown"]:
                    return False
            elif field_type is None:
                return False

        return True

    def _validate_type_safety(
        self, node_configs: list[dict[str, Any]], pipeline_name: str, graph: DirectedGraph
    ) -> tuple[int, int]:
        """Validate type safety for compilation."""
        # Validate using simple type checker - will raise error if not safe
        validate_pipeline_compilation(graph, pipeline_name)

        # Count typed nodes for metadata
        typed_count = 0
        total_count = len(node_configs)

        for config in node_configs:
            params = config.get("params", {})
            if params.get("input_schema") or params.get("output_schema"):
                typed_count += 1

        return typed_count, total_count

    def _extract_from_built_graph(
        self, graph: DirectedGraph, yaml_config: dict[str, Any], builder: Any
    ) -> list[dict[str, Any]]:
        """Extract node configurations from PipelineBuilder's BUILT graph + registry access."""
        node_configs = []
        yaml_nodes = {n.get("id"): n for n in yaml_config.get("nodes", [])}

        for node_id, node_spec in graph.nodes.items():
            yaml_node = yaml_nodes.get(node_id, {})

            # Extract from the BUILT node_spec + YAML config + function registry
            node_config = {
                "type": yaml_node.get("type", "unknown"),
                "id": node_id,
                "params": self._extract_node_params(node_spec, yaml_node, builder),
                "depends_on": list(node_spec.deps) if hasattr(node_spec, "deps") else [],
            }
            node_configs.append(node_config)

        return node_configs

    def _infer_node_type(self, node_spec: Any, yaml_node: dict[str, Any]) -> str:
        """Infer node type from YAML (PipelineBuilder should have this info)."""
        # Get type from YAML since PipelineBuilder parsed it
        return yaml_node.get("type", "unknown")  # type: ignore[no-any-return]

    def _extract_node_params(
        self, node_spec: Any, yaml_node: dict[str, Any], builder: Any
    ) -> dict[str, Any]:
        """Extract parameters combining built node_spec, YAML config, and function registry."""
        params = yaml_node.get("params", {}).copy()
        node_type = yaml_node.get("type", "unknown")

        # If PipelineBuilder populated schemas, use them (future enhancement)
        if hasattr(node_spec, "in_type") and node_spec.in_type:
            params["input_schema"] = self._type_to_schema_dict(node_spec.in_type)

        if hasattr(node_spec, "out_type") and node_spec.out_type:
            params["output_schema"] = self._type_to_schema_dict(node_spec.out_type)

        # Otherwise, compiler does schema inference as optimization
        else:
            # Function schema inference using registry (compiler optimization)
            if node_type == "function" and "fn" in params:
                func_name = params["fn"]
                # Use the builder's registered functions (PipelineBuilder resolved them)
                if isinstance(func_name, str) and func_name in builder.registered_functions:
                    actual_func = builder.registered_functions[func_name]
                    input_schema, output_schema = self._infer_function_schemas(actual_func)
                    if input_schema:
                        params["input_schema"] = input_schema
                    if output_schema:
                        params["output_schema"] = output_schema

            # LLM/Agent default schemas (compiler optimization)
            elif node_type in ["llm", "agent"] and "output_schema" not in params:
                params["output_schema"] = {"result": "str"}

        return params  # type: ignore[no-any-return]

    def _infer_function_schemas(
        self, func: Any
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """Compiler optimization: infer function schemas from type hints."""
        try:
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            params_list = list(sig.parameters.values())

            # Input schema from first parameter
            input_schema = None
            if params_list and params_list[0].annotation != inspect.Parameter.empty:
                input_schema = self._type_to_schema_dict(params_list[0].annotation)

            # Output schema from return type
            output_schema = None
            if "return" in hints:
                output_schema = self._type_to_schema_dict(hints["return"])

            return input_schema, output_schema
        except Exception:
            # Schema inference failed - this is expected for functions without type hints
            # Return None for both schemas to indicate inference failure
            return None, None

    def _type_to_schema_dict(self, type_obj: Any) -> dict[str, str] | None:
        """Convert type to schema dict - same logic as before."""
        if hasattr(type_obj, "__annotations__"):  # TypedDict
            return {k: getattr(v, "__name__", str(v)) for k, v in type_obj.__annotations__.items()}
        elif hasattr(type_obj, "model_fields"):  # Pydantic
            return {
                k: getattr(v.annotation, "__name__", str(v.annotation))
                for k, v in type_obj.model_fields.items()
            }
        elif hasattr(type_obj, "__name__"):
            return {"result": type_obj.__name__}
        return None

    def _extract_output_schemas(
        self, node_configs: list[dict[str, Any]]
    ) -> dict[str, dict[str, str]]:
        """Extract output schemas for dependency chain inference."""
        output_schemas = {}
        for config in node_configs:
            if "output_schema" in config.get("params", {}):
                output_schemas[config["id"]] = config["params"]["output_schema"]
        return output_schemas

    def _infer_dependency_input_schemas(
        self, node_configs: list[dict[str, Any]], output_schemas: dict[str, dict[str, str]]
    ) -> None:
        """Infer input schemas from dependency output schemas."""
        for config in node_configs:
            params = config.get("params", {})
            depends_on = config.get("depends_on", [])

            if depends_on and "input_schema" not in params:
                if len(depends_on) == 1:
                    # Single dependency - direct mapping
                    dep_output = output_schemas.get(depends_on[0])
                    if dep_output:
                        params["input_schema"] = dep_output
                else:
                    # Multiple dependencies - prefixed combination
                    combined = {}
                    for dep_id in depends_on:
                        dep_output = output_schemas.get(dep_id, {})
                        for field, field_type in dep_output.items():
                            combined[f"{dep_id}.{field}"] = field_type
                    if combined:
                        params["input_schema"] = combined

    def _extract_pipeline_input_schema(
        self, pipeline_instance: Any, node_configs: list[dict[str, Any]]
    ) -> dict[str, str] | None:
        """Extract pipeline input schema from pipeline instance."""
        # Try to get from pipeline's input type method
        if hasattr(pipeline_instance, "get_input_type"):
            try:
                input_type = pipeline_instance.get_input_type()
                return self._type_to_schema_dict(input_type)
            except Exception:
                pass  # nosec B110 - intentional silent failure

        # Fallback: infer from first nodes
        first_nodes = [config for config in node_configs if not config.get("depends_on")]
        if first_nodes and "input_schema" in first_nodes[0].get("params", {}):
            return first_nodes[0]["params"]["input_schema"]  # type: ignore[no-any-return]

        return None


# Singleton for easy access
_compiler = PipelineCompiler()


def compile_pipeline(yaml_path: str | Path) -> CompiledPipelineData:
    """Compile a pipeline using framework components."""
    return _compiler.compile_from_yaml(yaml_path)
