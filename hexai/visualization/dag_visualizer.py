"""DAG visualization using Graphviz for hexAI pipelines.

This module provides utilities to export DirectedGraph objects to Graphviz DOT format for
visualization and debugging purposes.
"""

import logging
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

from hexai.core.domain.dag import DirectedGraph

from .file_exporter import FileExporter
from .graph_renderer import GraphRenderer
from .schema_extractor import SchemaExtractor

logger = logging.getLogger(__name__)


class DAGVisualizer:
    """Main interface for DAG visualization - coordinates visualization components."""

    def __init__(self, graph: DirectedGraph):
        """Initialize visualizer with a DAG.

        Args:
            graph: The DirectedGraph to visualize
        """
        self.graph = graph
        self.renderer = GraphRenderer()
        self.schema_extractor = SchemaExtractor()
        self.exporter = FileExporter()

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

        Args:
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

        Returns:
            DOT format string for the graph
        """
        # Extract pipeline name if available
        pipeline_name = getattr(self.graph, "_pipeline_name", None)

        # Get compiled schema information
        compiled_schemas: dict[str, dict[str, Any]] = {}
        pipeline_input_schema = input_schema

        # Try to load compiled schema information first
        if pipeline_name and (
            show_node_schemas or show_intermediate_input or show_intermediate_output
        ):
            try:
                compiled_schemas, found_input_schema = self.schema_extractor.load_compiled_schemas(
                    pipeline_name
                )
                if found_input_schema and not pipeline_input_schema:
                    pipeline_input_schema = found_input_schema
            except Exception:
                # Compilation failed, use basic node information if available
                if basic_node_types:
                    compiled_schemas = {}
                    # Convert basic node information to compatible format
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

        # If no compiled schemas and we want to show schemas, try extracting from nodes
        if not compiled_schemas and show_node_schemas:
            compiled_schemas = self.schema_extractor.extract_all_schemas(self.graph)

        # Prepare options for renderer
        options = {
            "title": title,
            "node_attributes": node_attributes,
            "edge_attributes": edge_attributes,
            "show_io_nodes": show_io_nodes,
            "input_schema": pipeline_input_schema,
            "output_schema": output_schema,
            "show_intermediate_input": show_intermediate_input,
            "show_intermediate_output": show_intermediate_output,
            "basic_node_types": basic_node_types,
            "basic_node_schemas": basic_node_schemas,
        }

        # Generate DOT string
        return self.renderer.to_dot(self.graph, compiled_schemas, options)

    def render_to_file(
        self, output_path: str, format: str = "png", title: str = "Pipeline DAG", **kwargs: Any
    ) -> str:
        """Render DAG to file using Graphviz.

        Args:
            output_path: Path where to save the rendered graph (without extension)
            format: Output format ('png', 'svg', 'pdf', etc.)
            title: Title for the graph
            **kwargs: Additional arguments passed to to_dot()

        Returns:
            Path to the rendered file

        Raises:
            ImportError: If graphviz is not installed.
        """
        dot_string = self.to_dot(title=title, **kwargs)
        return self.exporter.render_to_file(dot_string, output_path, format)

    def show(self, title: str = "Pipeline DAG", **kwargs: Any) -> None:
        """Display DAG in default viewer.

        Args:
            title: Title for the graph
            **kwargs: Additional arguments passed to to_dot()

        Raises:
            RuntimeError: If showing graph fails.
        """
        dot_string = self.to_dot(title=title, **kwargs)
        self.exporter.show(dot_string, title)


# Backward compatibility functions
def export_dag_to_dot(
    graph: DirectedGraph,
    output_file: str | None = None,
    title: str = "Pipeline DAG",
    show_io_nodes: bool = True,
    input_schema: Any = None,
    output_schema: Any = None,
) -> str:
    """Export DAG to DOT format with I/O support.

    Args:
        graph: The DirectedGraph to export
        output_file: Optional file path to save DOT content
        title: Title for the graph
        show_io_nodes: Whether to show input/output nodes
        input_schema: Input schema information
        output_schema: Output schema information

    Returns:
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

    Args:
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

    Returns:
        Path to the rendered file
    """
    # Extract pipeline name from title if possible
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

    # Create Graphviz source and render
    dot = graphviz.Source(dot_content)
    return str(dot.render(output_path, format=format, cleanup=True))
