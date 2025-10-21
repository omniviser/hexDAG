"""DAG visualization using Graphviz for hexdag pipelines.

This module provides utilities to export DirectedGraph objects to Graphviz DOT format for
visualization and debugging purposes.
"""

import contextlib
import pathlib
import platform
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

try:
    import graphviz
except ImportError as e:
    raise ImportError(
        "Graphviz is not installed. Please install with:\n"
        "  pip install hexdag[viz]\n"
        "  or\n"
        "  uv pip install hexdag[viz]"
    ) from e

from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.logging import get_logger

logger = get_logger(__name__)


class DAGVisualizer:
    """Visualizes DirectedGraph objects using Graphviz."""

    def __init__(self, graph: DirectedGraph):
        """Initialize visualizer with a DAG.

        Args
        ----
            graph: The DirectedGraph to visualize
        """
        self.graph = graph
        self._dot = None

    def to_dot(
        self,
        title: str = "Pipeline DAG",
        node_attributes: dict[str, dict[str, Any]] | None = None,
        edge_attributes: dict[tuple[str, str], dict[str, Any]] | None = None,
        show_io_nodes: bool = True,
        input_schema: Any = None,
        output_schema: Any = None,
        _enhance_with_generated_schemas: bool = True,
        show_node_schemas: bool = True,
        show_intermediate_input: bool = False,
        show_intermediate_output: bool = False,
        basic_node_types: dict[str, str] | None = None,
        basic_node_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Export DAG to DOT format string with enhanced schema display options.

        Args
        ----
            title: Title for the graph
            node_attributes: Optional custom attributes for nodes
            edge_attributes: Optional custom attributes for edges
            show_io_nodes: Whether to show input/output nodes
            input_schema: Input schema information
            output_schema: Output schema information
            enhance_with_generated_schemas: Whether to try loading auto-generated schema files
            show_node_schemas: Whether to show input/output schemas on nodes
            show_intermediate_input: Whether to show input schemas on intermediate nodes
            show_intermediate_output: Whether to show output schemas on intermediate nodes
            basic_node_types: Basic node type information from YAML (fallback mode)
            basic_node_schemas: Basic schema information from YAML (fallback mode)

        Returns
        -------
            DOT format string for the graph
        """
        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir="TB", style="filled", bgcolor="white")
        dot.attr("node", shape="box", style="filled,rounded", fontname="Arial")
        dot.attr("edge", fontname="Arial")

        compiled_schemas: dict[str, dict[str, Any]] = {}
        pipeline_input_schema = input_schema

        # Try to load compiled schema information first
        pipeline_name = getattr(self.graph, "_pipeline_name", None)
        if pipeline_name and (
            show_node_schemas or show_intermediate_input or show_intermediate_output
        ):
            try:
                compiled_schemas, found_input_schema = self._load_compiled_schemas(pipeline_name)
                if found_input_schema and not pipeline_input_schema:
                    pipeline_input_schema = found_input_schema
            except Exception:
                # Compilation failed, use basic node information if available
                if basic_node_types:
                    compiled_schemas = {}
                    for node_id, node_type in basic_node_types.items():
                        node_schema_info = (
                            basic_node_schemas.get(node_id, {}) if basic_node_schemas else {}
                        )
                        compiled_schemas[node_id] = {
                            "node_type": node_type,
                            "input_schema": node_schema_info.get("input_schema"),
                            "output_schema": node_schema_info.get("output_schema"),
                        }
                        # Auto-assign default output for LLM/Agent nodes if not explicit
                        if node_type in ["llm", "agent"] and not node_schema_info.get(
                            "output_schema"
                        ):
                            compiled_schemas[node_id]["output_schema"] = {"result": "str"}

        first_nodes, last_nodes = self._find_terminal_nodes()

        if show_io_nodes and first_nodes:
            input_label = self._format_schema_label("🔵 PIPELINE INPUT", pipeline_input_schema)
            dot.node("__INPUT__", input_label, color="lightblue", fillcolor="lightblue")
            for first_node in first_nodes:
                dot.edge("__INPUT__", first_node)

        # Show pipeline output from final nodes
        if show_io_nodes and last_nodes:
            # Collect output schemas from all final nodes
            pipeline_output_schemas = {}
            for last_node in last_nodes:
                node_schemas = compiled_schemas.get(last_node, {})
                if node_schemas.get("output_schema"):
                    pipeline_output_schemas[last_node] = node_schemas["output_schema"]

            if pipeline_output_schemas:
                # If single output node, show its schema directly
                if len(pipeline_output_schemas) == 1:
                    output_node, output_schema_data = next(iter(pipeline_output_schemas.items()))
                    output_label = self._format_schema_label(
                        f"🟢 PIPELINE OUTPUT\\n({output_node})", output_schema_data
                    )
                else:
                    # Multiple output nodes - show combined
                    combined_output = {}
                    for node, schema in pipeline_output_schemas.items():
                        combined_output[f"{node}_output"] = schema
                    output_label = self._format_schema_label("🟢 PIPELINE OUTPUT", combined_output)
            elif output_schema:
                # Fallback to provided output schema
                output_label = self._format_schema_label("🟢 PIPELINE OUTPUT", output_schema)
            else:
                output_label = "🟢 PIPELINE OUTPUT"

            dot.node("__OUTPUT__", output_label, color="lightgreen", fillcolor="lightgreen")
            for last_node in last_nodes:
                dot.edge(last_node, "__OUTPUT__")

        for node_name, node_spec in self.graph.nodes.items():
            # Determine if this is an intermediate node
            is_first_node = node_name in first_nodes
            is_last_node = node_name in last_nodes
            is_intermediate = not (is_first_node and is_last_node)

            node_schemas = compiled_schemas.get(node_name, {})

            # Decide what schemas to show based on options and availability
            # Only show schemas when explicitly requested, not by default
            has_compiled_schemas = bool(
                node_schemas.get("input_schema") or node_schemas.get("output_schema")
            )

            if has_compiled_schemas:
                # Only show schemas when explicitly requested
                show_input_for_node = show_intermediate_input and is_intermediate
                show_output_for_node = show_intermediate_output and is_intermediate
            else:
                # Only show for intermediate nodes when explicitly requested (fallback mode)
                show_input_for_node = show_intermediate_input and is_intermediate
                show_output_for_node = show_intermediate_output and is_intermediate

            if (show_node_schemas or show_input_for_node or show_output_for_node) and node_schemas:
                input_schema_to_show = (
                    node_schemas.get("input_schema") if show_input_for_node else None
                )
                output_schema_to_show = (
                    node_schemas.get("output_schema") if show_output_for_node else None
                )

                label = self._create_enhanced_node_label(
                    node_name,
                    node_spec,
                    input_schema_to_show,
                    output_schema_to_show,
                    node_schemas.get("type"),
                    node_schemas.get("function_name"),
                )
            elif node_schemas and node_schemas.get("type"):
                # Show node type even without schemas when compiled data is available
                detected_node_type: str | None = node_schemas.get("type")
                function_name: str | None = node_schemas.get("function_name")

                if function_name:
                    label = f"📦 {node_name}\\n({detected_node_type}: {function_name})"
                else:
                    label = f"📦 {node_name}\\n({detected_node_type})"
            else:
                # Fallback to basic node label with node type if available
                basic_node_type: str | None = (
                    basic_node_types.get(node_name) if basic_node_types else None
                )
                if basic_node_type:
                    label = f"{node_name}\\n({basic_node_type})"
                else:
                    label = self._format_node_label(node_name, node_spec)

            # Apply custom attributes if provided
            node_attrs = node_attributes.get(node_name, {}) if node_attributes else {}

            styling_node_type: str | None = node_schemas.get("type") or (
                basic_node_types.get(node_name) if basic_node_types else None
            )
            default_attrs = self._get_node_style(node_spec, styling_node_type)
            default_attrs.update(node_attrs)

            dot.node(node_name, label, **default_attrs)

        for node_name, node_spec in self.graph.nodes.items():
            for dep in node_spec.deps:
                # Apply custom edge attributes if provided
                edge_key = (dep, node_name)
                edge_attrs = edge_attributes.get(edge_key, {}) if edge_attributes else {}
                dot.edge(dep, node_name, **edge_attrs)

        return dot.source  # type: ignore[no-any-return]

    def _extract_compiled_schemas(
        self, node_configs: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Extract schema information from compiled NODE_CONFIGS.

        Args
        ----
            node_configs: List of compiled node configurations

        Returns
        -------
            Dictionary mapping node_id to {input_schema, output_schema, node_type}
        """
        schemas = {}

        for node_config in node_configs:
            node_id = node_config.get("id")
            if not node_id:
                continue

            params = node_config.get("params", {})
            schemas[node_id] = {
                "input_schema": params.get("input_schema"),
                "output_schema": params.get("output_schema"),
                "type": node_config.get("type"),
                "function_name": (
                    params.get("fn") if node_config.get("type") == "function" else None
                ),
            }

        return schemas

    def _extract_node_input_schema(self, node_spec: Any) -> dict[str, str] | None:
        """Extract input schema information from a node specification.

        Args
        ----
            node_spec: Node specification object

        Returns
        -------
            Dictionary of input schema fields or None
        """
        if hasattr(node_spec, "in_model") and node_spec.in_model:
            return self._convert_type_to_schema_dict(node_spec.in_model)

        # Check for function-specific schema info
        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__annotations__"):
            return self._extract_function_input_schema(node_spec.fn)

        return None

    def _extract_node_output_schema(self, node_spec: Any) -> dict[str, str] | None:
        """Extract output schema information from a node specification.

        Args
        ----
            node_spec: Node specification object

        Returns
        -------
            Dictionary of output schema fields or None
        """
        if hasattr(node_spec, "out_model") and node_spec.out_model:
            return self._convert_type_to_schema_dict(node_spec.out_model)

        # Check for function-specific schema info
        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__annotations__"):
            return self._extract_function_output_schema(node_spec.fn)

        return None

    def _convert_type_to_schema_dict(self, type_obj: Any) -> dict[str, str] | None:
        """Convert a type object to a schema dictionary.

        Args
        ----
            type_obj: Type object to convert

        Returns
        -------
            Dictionary representation of the type
        """
        try:
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

            if isinstance(type_obj, dict):
                return type_obj

        except Exception:
            # Type conversion failed - this is expected for complex types
            pass  # nosec B110 - intentional silent failure for type conversion
        return None

    def _extract_function_input_schema(self, func: Any) -> dict[str, str] | None:
        """Extract input schema from function type hints.

        Args
        ----
            func: Function to analyze

        Returns
        -------
            Dictionary of input schema fields or None
        """
        try:
            import inspect
            from typing import get_type_hints

            hints = get_type_hints(func)
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            if params and params[0].name != "self":
                first_param = params[0]
                param_type = hints.get(first_param.name)
                return self._convert_type_to_schema_dict(param_type)

        except Exception:
            # Function signature analysis failed - this is expected for functions without type hints
            pass  # nosec B110 - intentional silent failure for function analysis
        return None

    def _extract_function_output_schema(self, func: Any) -> dict[str, str] | None:
        """Extract output schema from function return type hints.

        Args
        ----
            func: Function to analyze

        Returns
        -------
            Dictionary of output schema fields or None
        """
        try:
            from typing import get_type_hints

            hints = get_type_hints(func)
            return_type = hints.get("return")

            if return_type and return_type is not type(None):
                return self._convert_type_to_schema_dict(return_type)

        except Exception:
            pass  # nosec B110 - intentional silent failure for return type analysis
        return None

    def _create_enhanced_node_label(
        self,
        node_name: str,
        node_spec: Any,
        input_schema: dict[str, str] | None,
        output_schema: dict[str, str] | None,
        node_type: str | None = None,
        function_name: str | None = None,
    ) -> str:
        """Create an enhanced node label showing input/output schemas from compiled data.

        Args
        ----
            node_name: Name of the node
            node_spec: Node specification
            input_schema: Input schema dictionary from compiled data
            output_schema: Output schema dictionary from compiled data
            node_type: Node type from compiled data
            function_name: Function name from compiled data

        Returns
        -------
            Formatted label string for Graphviz
        """
        detected_type = node_type or getattr(node_spec, "type", "unknown")

        type_emoji = {
            "function": "⚙️",
            "llm": "🤖",
            "agent": "🧠",
            "loop": "🔄",
            "conditional": "🔀",
        }.get(str(detected_type) if detected_type else "unknown", "📦")

        # Start with node name and type
        label_parts = [f"{type_emoji} {node_name}"]

        if function_name:
            label_parts.append(f"({detected_type}: {function_name})")
        elif detected_type:
            label_parts.append(f"({detected_type})")

        if input_schema and input_schema != {"result": "Any"}:
            input_fields = []
            for field, field_type in input_schema.items():
                # Clean up type names
                clean_type = (
                    field_type.replace("typing.", "").replace("<class '", "").replace("'>", "")
                )
                input_fields.append(f"{field}: {clean_type}")

            if input_fields:
                if len(input_fields) <= 4:
                    input_str = "\\n".join(input_fields)
                else:
                    input_str = "\\n".join(input_fields[:4]) + "\\n..."
                label_parts.append(f"⬇️ IN\\n{input_str}")

        if output_schema and output_schema != {"result": "Any"}:
            output_fields = []
            for field, field_type in output_schema.items():
                # Clean up type names
                clean_type = (
                    field_type.replace("typing.", "").replace("<class '", "").replace("'>", "")
                )
                output_fields.append(f"{field}: {clean_type}")

            if output_fields:
                if len(output_fields) <= 4:
                    output_str = "\\n".join(output_fields)
                else:
                    output_str = "\\n".join(output_fields[:4]) + "\\n..."
                label_parts.append(f"⬆️ OUT\\n{output_str}")

        return "\\n\\n".join(label_parts)

    def _format_node_label(self, node_name: str, node_spec: Any) -> str:
        """Format a standard node label without schemas.

        Returns
        -------
            Formatted node label string.
        """
        node_type = getattr(node_spec, "type", "unknown")

        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__name__"):
            return f"📦 {node_name}\\n({node_type}: {node_spec.fn.__name__})"
        return f"📦 {node_name}\\n({node_type})"

    def _get_node_style(
        self, node_spec: Any, compiled_node_type: str | None = None
    ) -> dict[str, str]:
        """Get visual style for a node based on its type.

        Returns
        -------
            Dictionary of style attributes.
        """
        node_type = str(compiled_node_type or getattr(node_spec, "type", "unknown"))

        node_styles = {
            "function": {"color": "lightgreen", "fillcolor": "lightgreen"},
            "llm": {"color": "lightblue", "fillcolor": "lightblue"},
            "agent": {"color": "lightcoral", "fillcolor": "lightcoral"},
            "loop": {"color": "lightyellow", "fillcolor": "lightyellow"},
            "conditional": {"color": "lightpink", "fillcolor": "lightpink"},
        }
        return node_styles.get(node_type, {"color": "lightgray", "fillcolor": "lightgray"})

    def _find_io_nodes(self) -> tuple[list[str], list[str]]:
        """Find first nodes (no dependencies) and last nodes (no dependents).

        Returns
        -------
            Tuple of (first_nodes, last_nodes)
        """
        # Find first nodes (no dependencies)
        first_nodes = []
        for node_name in self.graph.nodes:
            dependencies = self.graph.get_dependencies(node_name)
            if not dependencies:
                first_nodes.append(node_name)

        # Find last nodes (no dependents)
        all_dependencies: set[str] = set()
        for node_name in self.graph.nodes:
            all_dependencies.update(self.graph.get_dependencies(node_name))

        last_nodes = [
            node_name for node_name in self.graph.nodes if node_name not in all_dependencies
        ]

        return first_nodes, last_nodes

    def _find_terminal_nodes(self) -> tuple[list[str], list[str]]:
        """Find first nodes (no dependencies) and last nodes (no dependents).

        Returns
        -------
            Tuple of (first_nodes, last_nodes)
        """
        # Find first nodes (no dependencies)
        first_nodes = []
        for node_name in self.graph.nodes:
            dependencies = self.graph.get_dependencies(node_name)
            if not dependencies:
                first_nodes.append(node_name)

        # Find last nodes (no dependents)
        all_dependencies: set[str] = set()
        for node_name in self.graph.nodes:
            all_dependencies.update(self.graph.get_dependencies(node_name))

        last_nodes = [
            node_name for node_name in self.graph.nodes if node_name not in all_dependencies
        ]

        return first_nodes, last_nodes

    def _format_schema_label(self, label: str, schema: Any) -> str:
        """Format a schema for display in a node label with enhanced Pydantic model support.

        Args
        ----
            label: Base label (INPUT/OUTPUT)
            schema: Schema information

        Returns
        -------
            Formatted label string
        """
        if schema is None:
            return label

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

            # Format for display
            if len(field_lines) <= 3:
                field_str = "\\n".join(field_lines)
            else:
                field_str = "\\n".join(field_lines[:3]) + "\\n..."

            return f"{label}\\n{schema.__name__}\\n{field_str}"

        if hasattr(schema, "__name__"):
            return f"{label}\\n({schema.__name__})"
        if hasattr(schema, "model_fields"):
            # Pydantic model instance
            fields = list(schema.model_fields.keys())
            field_str = ", ".join(fields) if len(fields) <= 3 else f"{', '.join(fields[:3])}..."
            return f"{label}\\n({field_str})"
        if isinstance(schema, dict):
            # Dict schema - format as field: type pairs for input primitives
            field_lines = []
            for key, value in schema.items():
                if isinstance(value, str):
                    # Input primitives format: {"field": "type"}
                    field_lines.append(f"{key}: {value}")
                else:
                    # Other dict formats
                    field_lines.append(key)

            if len(field_lines) <= 4:
                field_str = "\\n".join(field_lines)
            else:
                field_str = "\\n".join(field_lines[:4]) + "\\n..."
            return f"{label}\\n{field_str}"
        if isinstance(schema, type):
            # Type annotation
            return f"{label}\\n({schema.__name__})"
        # String representation
        schema_str = str(schema)
        if len(schema_str) > 30:
            schema_str = schema_str[:27] + "..."
        return f"{label}\\n({schema_str})"

    def _try_load_generated_schemas(self, pipeline_name: str, pipeline_dir: str) -> dict[str, Any]:
        """Try to load auto-generated schema file for enhanced visualization.

        Returns
        -------
            Dictionary of loaded schemas, empty dict if loading fails.
        """
        try:
            import importlib.util
            from pathlib import Path

            schema_file = Path(pipeline_dir) / f"{pipeline_name}_schemas.py"

            if not schema_file.exists():
                return {}

            # Dynamically load the schema module
            spec = importlib.util.spec_from_file_location(f"{pipeline_name}_schemas", schema_file)
            if spec and spec.loader:
                schema_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(schema_module)

                schemas = {}
                for attr_name in dir(schema_module):
                    attr = getattr(schema_module, attr_name)
                    if (
                        isinstance(attr, type)
                        and hasattr(attr, "model_fields")
                        and attr_name.endswith(("Input", "Output"))
                    ):
                        schemas[attr_name] = attr

                return schemas

        except Exception:
            # Silently fail if schema loading doesn't work
            pass  # nosec B110 - intentional silent failure for schema loading
        return {}

    def _load_compiled_schemas(
        self, pipeline_name: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None]:
        """Load schema information by compiling the pipeline on-the-fly.

        Instead of reading pre-compiled files, this now compiles the pipeline
        in memory to extract all type information for visualization.
        Handles compilation failures gracefully.

        Args
        ----
            pipeline_name: Name of the pipeline

        Returns
        -------
            Tuple of (node_schemas_dict, pipeline_input_schema)
        """
        try:
            # Note: Pipeline compiler has been removed in favor of simple caching.
            # Schema information should be provided via basic_node_schemas parameter
            # or extracted from runtime Config classes.
            logger.debug(
                "Schema visualization - compiler removed, use basic_node_schemas parameter"
            )
            return {}, None

        except Exception as e:
            # Silently fail - schemas are optional for visualization
            logger.debug("Exception in schema loading: %s", e)
            return {}, None

    def _get_node_attributes(
        self,
        node_name: str,
        custom_attributes: dict[str, dict[str, Any]] | None = None,
        generated_schemas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get attributes for a node with enhanced schema information.

        Returns
        -------
            Dictionary of node attributes.
        """
        node_spec = self.graph.nodes[node_name]
        generated_schemas = generated_schemas or {}

        # Basic attributes
        attrs = {"label": node_name, "fontname": "Arial", "fontsize": "10"}

        # Enhanced type information with generated schemas
        if node_spec.in_model or node_spec.out_model or generated_schemas:
            in_name = (
                getattr(node_spec.in_model, "__name__", "Any") if node_spec.in_model else "Any"
            )
            out_name = (
                getattr(node_spec.out_model, "__name__", "Any") if node_spec.out_model else "Any"
            )

            # Check for enhanced schema names from generated files
            for schema_name, schema_class in generated_schemas.items():
                if f"{node_name.title().replace('_', '')}Input" in schema_name:
                    in_name = schema_class.__name__
                elif f"{node_name.title().replace('_', '')}Output" in schema_name:
                    out_name = schema_class.__name__

            attrs["label"] = f"{node_name}\\n({in_name} → {out_name})"

        # Enhanced coloring based on schema complexity
        has_complex_schema = (
            any(node_name.lower() in schema_name.lower() for schema_name in generated_schemas)
            if generated_schemas
            else False
        )

        # Color based on function type
        fn_name = getattr(node_spec.fn, "__name__", str(node_spec.fn))
        if "llm" in fn_name.lower():
            attrs["fillcolor"] = "lightblue"
            attrs["style"] = "filled"
        elif "agent" in fn_name.lower():
            attrs["fillcolor"] = "lightgreen"
            attrs["style"] = "filled"
        elif "tool" in fn_name.lower():
            attrs["fillcolor"] = "lightyellow"
            attrs["style"] = "filled"

        # Highlight nodes with generated schemas
        if has_complex_schema:
            attrs["style"] = attrs.get("style", "filled") + ",bold"
            attrs["penwidth"] = "2"

        if custom_attributes and node_name in custom_attributes:
            attrs.update(custom_attributes[node_name])

        return attrs

    def _get_edge_attributes(
        self,
        edge: tuple[str, str],
        custom_attributes: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Get attributes for an edge.

        Returns
        -------
            Dictionary of edge attributes.
        """
        attrs = {"fontname": "Arial", "fontsize": "8"}

        if custom_attributes and edge in custom_attributes:
            attrs.update(custom_attributes[edge])

        return attrs

    def _format_attributes(self, attrs: dict[str, Any]) -> str:
        """Format attributes for DOT notation.

        Returns
        -------
            Formatted attribute string.
        """
        if not attrs:
            return ""

        attr_pairs = []
        for key, value in attrs.items():
            # Escape quotes in values
            if isinstance(value, str):
                value = value.replace('"', '\\"')
                attr_pairs.append(f'{key}="{value}"')
            else:
                attr_pairs.append(f"{key}={value}")

        return f"[{', '.join(attr_pairs)}]"

    def render_to_file(
        self, output_path: str, format: str = "png", title: str = "Pipeline DAG", **kwargs: Any
    ) -> str:
        """Render DAG to file using Graphviz.

        Args
        ----
            output_path: Path where to save the rendered graph (without extension)
            format: Output format ('png', 'svg', 'pdf', etc.)
            title: Title for the graph
            **kwargs: Additional arguments passed to to_dot()

        Returns
        -------
            Path to the rendered file

        Raises
        ------
        ImportError
            If graphviz is not installed.
        RuntimeError
            If rendering fails.
        """
        dot_string = self.to_dot(title=title, **kwargs)

        # Use subprocess to avoid Source.gv creation
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as temp_file:
                temp_file.write(dot_string)
                temp_dot_path = temp_file.name

            # Use dot command to render
            output_file = f"{output_path}.{format}"
            # nosec B607, B603 - dot is a trusted system command for Graphviz
            subprocess.run(  # nosec B607, B603
                ["dot", "-T" + format, "-o", output_file, temp_dot_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Clean up temporary file
            with contextlib.suppress(OSError):
                pathlib.Path(temp_dot_path).unlink()

            return output_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to render graph: {e.stderr}") from e
        except FileNotFoundError:
            raise ImportError(
                "Graphviz 'dot' command not found. Please install Graphviz."
            ) from None
        except Exception as e:
            raise RuntimeError(f"Failed to render graph: {e}") from e

    def show(self, title: str = "Pipeline DAG", **kwargs: Any) -> None:
        """Display DAG in default viewer.

        Args
        ----
            title: Title for the graph
            **kwargs: Additional arguments passed to to_dot()

        Raises
        ------
        RuntimeError
            If showing graph fails.
        """
        dot_string = self.to_dot(title=title, **kwargs)

        # Use subprocess to avoid Source.gv creation
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as temp_file:
                temp_file.write(dot_string)
                temp_dot_path = temp_file.name

            # Use dot command to create a temporary image and open it
            temp_image_path = temp_dot_path.replace(".dot", ".png")
            # nosec B607, B603 - dot is a trusted system command for Graphviz
            subprocess.run(  # nosec B607, B603
                ["dot", "-Tpng", "-o", temp_image_path, temp_dot_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Open the image with the default viewer
            # nosec B607, B603 - open is a trusted system command for viewing files
            system_platform = platform.system()
            if system_platform == "Darwin":
                viewer_cmd = "open"
            elif system_platform == "Linux":
                viewer_cmd = "xdg-open"
            else:
                viewer_cmd = None

            if viewer_cmd and shutil.which(viewer_cmd):
                subprocess.run([viewer_cmd, temp_image_path], check=False)  # nosec B607, B603
            else:
                help_msg = (
                    f"No default image viewer found for platform '{system_platform}'.\n"
                    f"For macOS, please ensure the 'open' command is available.\n"
                    f"For Linux, please ensure the 'xdg-open' command is installed.\n"
                    f"You can manually open the file located at: {temp_image_path}"
                )
                logger.error(help_msg)

            def cleanup_files() -> None:
                time.sleep(2)  # Wait for viewer to open
                try:
                    pathlib.Path(temp_dot_path).unlink()
                    pathlib.Path(temp_image_path).unlink()
                except OSError:
                    pass

            threading.Thread(target=cleanup_files, daemon=True).start()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to show graph: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to show graph: {e}") from e


def export_dag_to_dot(
    graph: DirectedGraph,
    output_file: str | None = None,
    title: str = "Pipeline DAG",
    show_io_nodes: bool = True,
    input_schema: Any = None,
    output_schema: Any = None,
) -> str:
    """Export DAG to DOT format with I/O support.

    Args
    ----
        graph: The DirectedGraph to export
        output_file: Optional file path to save DOT content
        title: Title for the graph
        show_io_nodes: Whether to show input/output nodes
        input_schema: Input schema information
        output_schema: Output schema information

    Returns
    -------
        DOT format string
    """
    visualizer = DAGVisualizer(graph)
    dot_string = visualizer.to_dot(
        title=title,
        show_io_nodes=show_io_nodes,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    if output_file:
        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(dot_string)

    return dot_string


def render_dag_to_image(
    graph: DirectedGraph,
    output_path: str,
    format: str = "png",
    title: str = "Pipeline DAG",
    show_io_nodes: bool = True,
    input_schema: Any = None,
    output_schema: Any = None,
    show_node_schemas: bool = True,
    show_intermediate_input: bool = False,
    show_intermediate_output: bool = False,
    basic_node_types: dict[str, str] | None = None,
    basic_node_schemas: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render DAG to image file with enhanced schema and intermediate node support.

    Args
    ----
        graph: The DirectedGraph to render
        output_path: Path where to save the rendered graph (without extension)
        format: Output format ('png', 'svg', 'pdf', etc.)
        title: Title for the graph
        show_io_nodes: Whether to show input/output nodes
        input_schema: Input schema information
        output_schema: Output schema information
        show_node_schemas: Whether to show schemas on nodes
        show_intermediate_input: Whether to show input schemas on intermediate nodes
        show_intermediate_output: Whether to show output schemas on intermediate nodes
        basic_node_types: Basic node type information from YAML (fallback mode)
        basic_node_schemas: Basic schema information from YAML (fallback mode)

    Returns
    -------
        Path to the rendered file
    """
    if "Pipeline:" in title and not hasattr(graph, "_pipeline_name"):
        pipeline_name = title.split("Pipeline:")[-1].strip()
        object.__setattr__(graph, "_pipeline_name", pipeline_name)

    visualizer = DAGVisualizer(graph)

    # Generate and render the DOT content with enhanced options
    dot_content = visualizer.to_dot(
        title=title,
        show_io_nodes=show_io_nodes,
        input_schema=input_schema,
        output_schema=output_schema,
        show_node_schemas=show_node_schemas,
        show_intermediate_input=show_intermediate_input,
        show_intermediate_output=show_intermediate_output,
        basic_node_types=basic_node_types,
        basic_node_schemas=basic_node_schemas,
    )

    dot = graphviz.Source(dot_content)
    return str(dot.render(output_path, format=format, cleanup=True))
