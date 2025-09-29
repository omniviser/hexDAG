"""Graph rendering and DOT generation for DAG visualization."""

import logging
from typing import Any

import graphviz

from hexai.core.domain.dag import DirectedGraph

logger = logging.getLogger(__name__)


class GraphRenderer:
    """Handles DOT format generation and node/edge styling."""

    def __init__(self) -> None:
        """Initialize the graph renderer."""
        self.node_styles = {
            "function": {"color": "lightgreen", "fillcolor": "lightgreen"},
            "llm": {"color": "lightblue", "fillcolor": "lightblue"},
            "agent": {"color": "lightcoral", "fillcolor": "lightcoral"},
            "loop": {"color": "lightyellow", "fillcolor": "lightyellow"},
            "conditional": {"color": "lightpink", "fillcolor": "lightpink"},
        }

    def to_dot(
        self,
        graph: DirectedGraph,
        schemas: dict[str, dict[str, Any]],
        options: dict[str, Any],
    ) -> str:
        """Generate DOT format string for the graph.

        Args:
            graph: The DirectedGraph to render
            schemas: Schema information for nodes
            options: Rendering options including title, show_io_nodes, etc.

        Returns:
            DOT format string
        """
        title = options.get("title", "Pipeline DAG")
        show_io_nodes = options.get("show_io_nodes", True)
        show_intermediate_input = options.get("show_intermediate_input", False)
        show_intermediate_output = options.get("show_intermediate_output", False)
        node_attributes = options.get("node_attributes", {})
        edge_attributes = options.get("edge_attributes", {})
        basic_node_types = options.get("basic_node_types", {})

        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir="TB", style="filled", bgcolor="white")
        dot.attr("node", shape="box", style="filled,rounded", fontname="Arial")
        dot.attr("edge", fontname="Arial")

        # Find terminal nodes
        first_nodes, last_nodes = self.find_terminal_nodes(graph)

        # Handle input/output nodes
        if show_io_nodes:
            self._add_io_nodes(dot, first_nodes, last_nodes, schemas, options)

        # Add regular nodes
        for node_name, node_spec in graph.nodes.items():
            label = self._create_node_label(
                node_name,
                node_spec,
                schemas.get(node_name, {}),
                first_nodes,
                last_nodes,
                show_intermediate_input,
                show_intermediate_output,
                basic_node_types.get(node_name) if basic_node_types else None,
            )

            node_attrs = node_attributes.get(node_name, {}) if node_attributes else {}
            node_type = schemas.get(node_name, {}).get("type") or (
                basic_node_types.get(node_name) if basic_node_types else None
            )
            default_attrs = self.get_node_style(node_type)
            default_attrs.update(node_attrs)

            dot.node(node_name, label, **default_attrs)

        # Add edges
        for node_name, node_spec in graph.nodes.items():
            for dep in node_spec.deps:
                edge_key = (dep, node_name)
                edge_attrs = edge_attributes.get(edge_key, {}) if edge_attributes else {}
                dot.edge(dep, node_name, **edge_attrs)

        return str(dot.source)

    def _add_io_nodes(
        self,
        dot: graphviz.Digraph,
        first_nodes: list[str],
        last_nodes: list[str],
        schemas: dict[str, dict[str, Any]],
        options: dict[str, Any],
    ) -> None:
        """Add input and output nodes to the graph.

        Args:
            dot: Graphviz Digraph object
            first_nodes: List of first nodes (no dependencies)
            last_nodes: List of last nodes (no dependents)
            schemas: Schema information for nodes
            options: Rendering options
        """
        input_schema = options.get("input_schema")
        output_schema = options.get("output_schema")

        # Add input node
        if first_nodes:
            input_label = self.format_schema_label("🔵 PIPELINE INPUT", input_schema)
            dot.node("__INPUT__", input_label, color="lightblue", fillcolor="lightblue")
            for first_node in first_nodes:
                dot.edge("__INPUT__", first_node)

        # Add output node
        if last_nodes:
            # Collect output schemas from final nodes
            pipeline_output_schemas = {}
            for last_node in last_nodes:
                node_schemas = schemas.get(last_node, {})
                if node_schemas.get("output_schema"):
                    pipeline_output_schemas[last_node] = node_schemas["output_schema"]

            if pipeline_output_schemas:
                if len(pipeline_output_schemas) == 1:
                    output_node, output_schema_data = next(iter(pipeline_output_schemas.items()))
                    output_label = self.format_schema_label(
                        f"🟢 PIPELINE OUTPUT\\n({output_node})", output_schema_data
                    )
                else:
                    combined_output = {}
                    for node, schema in pipeline_output_schemas.items():
                        combined_output[f"{node}_output"] = schema
                    output_label = self.format_schema_label("🟢 PIPELINE OUTPUT", combined_output)
            elif output_schema:
                output_label = self.format_schema_label("🟢 PIPELINE OUTPUT", output_schema)
            else:
                output_label = "🟢 PIPELINE OUTPUT"

            dot.node("__OUTPUT__", output_label, color="lightgreen", fillcolor="lightgreen")
            for last_node in last_nodes:
                dot.edge(last_node, "__OUTPUT__")

    def _create_node_label(
        self,
        node_name: str,
        node_spec: Any,
        node_schemas: dict[str, Any],
        first_nodes: list[str],
        last_nodes: list[str],
        show_intermediate_input: bool,
        show_intermediate_output: bool,
        basic_node_type: str | None = None,
    ) -> str:
        """Create a label for a node.

        Args:
            node_name: Name of the node
            node_spec: Node specification
            node_schemas: Schema information for the node
            first_nodes: List of first nodes
            last_nodes: List of last nodes
            show_intermediate_input: Whether to show input schemas
            show_intermediate_output: Whether to show output schemas
            basic_node_type: Basic node type from YAML

        Returns:
            Formatted label string
        """
        is_intermediate = not (node_name in first_nodes and node_name in last_nodes)
        has_compiled_schemas = bool(
            node_schemas.get("input_schema") or node_schemas.get("output_schema")
        )

        show_input = show_intermediate_input and is_intermediate and has_compiled_schemas
        show_output = show_intermediate_output and is_intermediate and has_compiled_schemas

        if show_input or show_output:
            return self.create_enhanced_node_label(
                node_name,
                node_spec,
                node_schemas.get("input_schema") if show_input else None,
                node_schemas.get("output_schema") if show_output else None,
                node_schemas.get("type"),
                node_schemas.get("function_name"),
            )
        if node_schemas.get("type"):
            detected_type = node_schemas.get("type")
            function_name = node_schemas.get("function_name")
            if function_name:
                return f"📦 {node_name}\\n({detected_type}: {function_name})"
            return f"📦 {node_name}\\n({detected_type})"
        if basic_node_type:
            return f"{node_name}\\n({basic_node_type})"
        return self._format_simple_node_label(node_name, node_spec)

    def create_enhanced_node_label(
        self,
        node_name: str,
        node_spec: Any,
        input_schema: dict[str, str] | None,
        output_schema: dict[str, str] | None,
        node_type: str | None = None,
        function_name: str | None = None,
    ) -> str:
        """Create an enhanced node label with schemas.

        Args:
            node_name: Name of the node
            node_spec: Node specification
            input_schema: Input schema dictionary
            output_schema: Output schema dictionary
            node_type: Node type
            function_name: Function name for function nodes

        Returns:
            Formatted label string
        """
        detected_type = node_type or getattr(node_spec, "type", "unknown")

        type_emoji = {
            "function": "⚙️",
            "llm": "🤖",
            "agent": "🧠",
            "loop": "🔄",
            "conditional": "🔀",
        }.get(str(detected_type), "📦")

        label_parts = [f"{type_emoji} {node_name}"]

        if function_name:
            label_parts.append(f"({detected_type}: {function_name})")
        elif detected_type:
            label_parts.append(f"({detected_type})")

        # Add input schema
        if input_schema and input_schema != {"result": "Any"}:
            input_fields = []
            for field, field_type in input_schema.items():
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

        # Add output schema
        if output_schema and output_schema != {"result": "Any"}:
            output_fields = []
            for field, field_type in output_schema.items():
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

    def _format_simple_node_label(self, node_name: str, node_spec: Any) -> str:
        """Format a simple node label without schemas.

        Args:
            node_name: Name of the node
            node_spec: Node specification

        Returns:
            Formatted label string
        """
        node_type = getattr(node_spec, "type", "unknown")

        if hasattr(node_spec, "fn") and hasattr(node_spec.fn, "__name__"):
            return f"📦 {node_name}\\n({node_type}: {node_spec.fn.__name__})"
        return f"📦 {node_name}\\n({node_type})"

    def get_node_style(self, node_type: str | None) -> dict[str, str]:
        """Get visual style for a node based on its type.

        Args:
            node_type: Type of the node

        Returns:
            Dictionary of style attributes
        """
        if node_type:
            return self.node_styles.get(
                str(node_type), {"color": "lightgray", "fillcolor": "lightgray"}
            )
        return {"color": "lightgray", "fillcolor": "lightgray"}

    def get_edge_style(self, source: str, target: str) -> dict[str, str]:
        """Get edge styling attributes.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            Dictionary of edge style attributes
        """
        return {"fontname": "Arial", "fontsize": "8"}

    def find_terminal_nodes(self, graph: DirectedGraph) -> tuple[list[str], list[str]]:
        """Find first nodes (no dependencies) and last nodes (no dependents).

        Args:
            graph: The DirectedGraph to analyze

        Returns:
            Tuple of (first_nodes, last_nodes)
        """
        first_nodes = []
        for node_name in graph.nodes:
            dependencies = graph.get_dependencies(node_name)
            if not dependencies:
                first_nodes.append(node_name)

        all_dependencies = set()
        for node_name in graph.nodes:
            all_dependencies.update(graph.get_dependencies(node_name))

        last_nodes = [node_name for node_name in graph.nodes if node_name not in all_dependencies]

        return first_nodes, last_nodes

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

    def format_attributes(self, attrs: dict[str, Any]) -> str:
        """Format attributes for DOT notation.

        Args:
            attrs: Dictionary of attributes

        Returns:
            Formatted attribute string
        """
        if not attrs:
            return ""

        attr_pairs = []
        for key, value in attrs.items():
            if isinstance(value, str):
                value = value.replace('"', '\\"')
                attr_pairs.append(f'{key}="{value}"')
            else:
                attr_pairs.append(f"{key}={value}")

        return f"[{', '.join(attr_pairs)}]"
