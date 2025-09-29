"""Tests for DAGVisualizer class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from hexai.core.domain.dag import DirectedGraph, NodeSpec

# Skip tests if visualization is not available
pytest.importorskip("graphviz")

from hexai.visualization import DAGVisualizer
from hexai.visualization.dag_visualizer import export_dag_to_dot, render_dag_to_image


class TestInput(BaseModel):
    """Test input model."""

    text: str
    priority: int


class TestOutput(BaseModel):
    """Test output model."""

    result: str
    confidence: float


def create_test_function():
    """Create a test function for testing."""
    return lambda x, context: {"result": "test", "confidence": 0.8}


class TestDAGVisualizer:
    """Tests for DAGVisualizer class."""

    def test_initialization(self):
        """Test DAGVisualizer initialization."""
        graph = DirectedGraph()
        visualizer = DAGVisualizer(graph)

        assert visualizer.graph == graph
        # Check for new component architecture
        assert visualizer.renderer is not None
        assert visualizer.schema_extractor is not None
        assert visualizer.exporter is not None

    def test_to_dot_basic(self):
        """Test basic DOT generation."""
        graph = DirectedGraph()
        node1 = NodeSpec("input", create_test_function())
        node2 = NodeSpec("process", create_test_function()).after("input")
        node3 = NodeSpec("output", create_test_function()).after("process")

        graph.add_many(node1, node2, node3)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot(title="Test Graph")

        # Check basic structure
        assert "digraph" in dot_string
        assert "Test Graph" in dot_string
        assert "input" in dot_string
        assert "process" in dot_string
        assert "output" in dot_string
        assert "input -> process" in dot_string
        assert "process -> output" in dot_string

    def test_to_dot_with_io_nodes(self):
        """Test DOT generation with I/O nodes."""
        graph = DirectedGraph()
        node = NodeSpec("process", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot(
            title="I/O Test",
            show_io_nodes=True,
            input_schema=TestInput,
            output_schema=TestOutput,
        )

        # Check for I/O nodes
        assert "__INPUT__" in dot_string
        assert "__OUTPUT__" in dot_string
        assert "PIPELINE INPUT" in dot_string
        assert "PIPELINE OUTPUT" in dot_string
        assert "__INPUT__ -> process" in dot_string
        assert "process -> __OUTPUT__" in dot_string

    def test_to_dot_with_custom_attributes(self):
        """Test DOT generation with custom node and edge attributes."""
        graph = DirectedGraph()
        node1 = NodeSpec("node1", create_test_function())
        node2 = NodeSpec("node2", create_test_function()).after("node1")
        graph.add_many(node1, node2)

        visualizer = DAGVisualizer(graph)

        node_attrs = {"node1": {"color": "red"}}
        edge_attrs = {("node1", "node2"): {"style": "dashed"}}

        dot_string = visualizer.to_dot(
            title="Custom Attributes", node_attributes=node_attrs, edge_attributes=edge_attrs
        )

        assert "Custom Attributes" in dot_string
        assert "node1" in dot_string
        assert "node2" in dot_string

    def test_to_dot_with_schemas(self):
        """Test DOT generation with schema display."""
        graph = DirectedGraph()
        node = NodeSpec("processor", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)

        # Test with show_node_schemas enabled
        dot_string = visualizer.to_dot(
            title="Schema Test",
            show_node_schemas=True,
            input_schema={"input": "str"},
            output_schema={"output": "str"},
        )

        assert "processor" in dot_string

    def test_to_dot_with_intermediate_schemas(self):
        """Test DOT generation with intermediate node schemas."""
        graph = DirectedGraph()
        node1 = NodeSpec("node1", create_test_function())
        node2 = NodeSpec("node2", create_test_function()).after("node1")
        graph.add_many(node1, node2)

        visualizer = DAGVisualizer(graph)

        dot_string = visualizer.to_dot(
            title="Intermediate Schema Test",
            show_intermediate_input=True,
            show_intermediate_output=True,
        )

        # Should show intermediate node schemas
        assert "node1" in dot_string
        assert "node2" in dot_string

    def test_format_schema_label(self):
        """Test schema label formatting."""
        graph = DirectedGraph()
        visualizer = DAGVisualizer(graph)

        # Test with Pydantic model
        schema = TestInput
        label = visualizer.renderer.format_schema_label("Input", schema)
        assert "text: str" in label
        assert "priority: int" in label

        # Test with dict schema
        schema_dict = {"text": "str", "priority": "int"}
        label = visualizer.renderer.format_schema_label("Input", schema_dict)
        assert "text: str" in label
        assert "priority: int" in label

        # Test with string schema
        label = visualizer.renderer.format_schema_label("Input", "str")
        assert "str" in label

    def test_get_node_style(self):
        """Test node style determination."""
        graph = DirectedGraph()
        node = NodeSpec("test_node", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)

        # Test with function type
        style = visualizer.renderer.get_node_style("function")
        assert "fillcolor" in style

        # Test with LLM type
        style = visualizer.renderer.get_node_style("llm")
        assert "fillcolor" in style

        # Test with agent type
        style = visualizer.renderer.get_node_style("agent")
        assert "fillcolor" in style

        # Test with unknown type
        style = visualizer.renderer.get_node_style("unknown")
        assert "fillcolor" in style

    def test_find_io_nodes(self):
        """Test finding input/output nodes."""
        graph = DirectedGraph()
        node1 = NodeSpec("input", create_test_function())
        node2 = NodeSpec("process", create_test_function()).after("input")
        node3 = NodeSpec("output", create_test_function()).after("process")

        graph.add_many(node1, node2, node3)

        visualizer = DAGVisualizer(graph)
        # This functionality is now in GraphRenderer
        input_nodes, output_nodes = visualizer.renderer.find_terminal_nodes(graph)

        assert "input" in input_nodes
        assert "output" in output_nodes

    def test_find_terminal_nodes(self):
        """Test finding terminal nodes."""
        graph = DirectedGraph()
        node1 = NodeSpec("start", create_test_function())
        node2 = NodeSpec("middle", create_test_function()).after("start")
        node3 = NodeSpec("end", create_test_function()).after("middle")

        graph.add_many(node1, node2, node3)

        visualizer = DAGVisualizer(graph)
        sources, sinks = visualizer.renderer.find_terminal_nodes(graph)

        assert "start" in sources
        assert "end" in sinks

    @patch("subprocess.run")
    def test_render_to_file(self, mock_subprocess):
        """Test rendering DAG to file."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        mock_subprocess.return_value.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"
            result = visualizer.render_to_file(output_path, format="png")

            assert result == f"{output_path}.png"

    def test_show_method(self):
        """Test show method with mocked subprocess."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)

        # This should not raise
        with patch("subprocess.run"):
            with patch("threading.Thread"):
                visualizer.show(title="Test Display")

    def test_export_dag_to_dot(self):
        """Test backward compatibility function."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = f"{tmpdir}/test.dot"
            dot_string = export_dag_to_dot(graph, output_file=output_file)

            assert "digraph" in dot_string
            assert Path(output_file).exists()

    @patch("graphviz.Source")
    def test_render_dag_to_image(self, mock_source):
        """Test backward compatibility image rendering."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        mock_dot_obj = Mock()
        mock_source.return_value = mock_dot_obj
        mock_dot_obj.render.return_value = "test.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"
            result = render_dag_to_image(graph, output_path, format="png")

            assert result == "test.png"

    @patch("graphviz.Source")
    def test_render_dag_to_image_with_schemas(self, mock_source):
        """Test rendering with schema display options."""
        graph = DirectedGraph()
        node1 = NodeSpec("input", create_test_function())
        node2 = NodeSpec("output", create_test_function()).after("input")
        graph.add_many(node1, node2)

        mock_dot_obj = Mock()
        mock_source.return_value = mock_dot_obj
        mock_dot_obj.render.return_value = "test.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"
            result = render_dag_to_image(
                graph,
                output_path,
                format="png",
                show_node_schemas=True,
                show_intermediate_input=True,
                show_intermediate_output=True,
            )

            assert result == "test.png"

    @patch("graphviz.Source")
    def test_render_dag_to_image_with_basic_info(self, mock_source):
        """Test rendering with basic node information."""
        graph = DirectedGraph()
        node = NodeSpec("processor", create_test_function())
        graph.add(node)

        mock_dot_obj = Mock()
        mock_source.return_value = mock_dot_obj
        mock_dot_obj.render.return_value = "test.png"

        basic_node_types = {"processor": "function"}
        basic_node_schemas = {
            "processor": {"input_schema": {"data": "str"}, "output_schema": {"result": "str"}}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"
            result = render_dag_to_image(
                graph,
                output_path,
                format="png",
                basic_node_types=basic_node_types,
                basic_node_schemas=basic_node_schemas,
            )

            assert result == "test.png"

    def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = DirectedGraph()
        visualizer = DAGVisualizer(graph)

        dot_string = visualizer.to_dot(title="Empty Graph")

        assert "digraph" in dot_string
        assert "Empty Graph" in dot_string

    def test_single_node_graph(self):
        """Test single node graph visualization."""
        graph = DirectedGraph()
        node = NodeSpec("single", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        assert "single" in dot_string

    def test_cyclic_graph_handling(self):
        """Test that visualizer can handle graphs without cycles."""
        # DirectedGraph doesn't allow cycles by design
        # This test verifies that valid DAGs work correctly
        graph = DirectedGraph()
        node1 = NodeSpec("node1", create_test_function())
        node2 = NodeSpec("node2", create_test_function()).after("node1")
        node3 = NodeSpec("node3", create_test_function()).after("node1", "node2")

        # This should work fine (it's a valid DAG)
        graph.add_many(node1, node2, node3)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        # Verify the DAG structure is correctly represented
        assert "node1 -> node2" in dot_string
        assert "node1 -> node3" in dot_string
        assert "node2 -> node3" in dot_string

    def test_node_with_complex_params(self):
        """Test node with complex parameters."""
        graph = DirectedGraph()

        def complex_func(x, context, **kwargs):
            return {"result": str(kwargs)}
        node = NodeSpec("complex", complex_func, params={"option1": "value1", "option2": 42})
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        assert "complex" in dot_string

    def test_node_with_none_schemas(self):
        """Test nodes with None schemas."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot(input_schema=None, output_schema=None)

        assert "test" in dot_string

    def test_large_graph_performance(self):
        """Test performance with larger graph."""
        graph = DirectedGraph()

        # Create a larger graph
        for i in range(20):
            if i == 0:
                node = NodeSpec(f"node_{i}", create_test_function())
            else:
                node = NodeSpec(f"node_{i}", create_test_function()).after(f"node_{i - 1}")
            graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        # Should contain all nodes
        for i in range(20):
            assert f"node_{i}" in dot_string

    def test_special_characters_in_node_names(self):
        """Test handling of special characters in node names."""
        graph = DirectedGraph()
        node = NodeSpec("test-node_1.2", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        assert "test-node_1.2" in dot_string

    def test_unicode_characters_in_node_names(self):
        """Test handling of Unicode characters in node names."""
        graph = DirectedGraph()
        node = NodeSpec("test_节点", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot()

        assert "test_节点" in dot_string

    # Integration tests for refactored components
    def test_visualizer_components_initialization(self):
        """Test that all refactored components are properly initialized."""
        graph = DirectedGraph()
        visualizer = DAGVisualizer(graph)

        assert hasattr(visualizer, "renderer")
        assert hasattr(visualizer, "schema_extractor")
        assert hasattr(visualizer, "exporter")

    def test_components_interaction(self):
        """Test that refactored components work together properly."""
        graph = DirectedGraph()
        node1 = NodeSpec("input", create_test_function())
        node2 = NodeSpec("process", create_test_function()).after("input")
        graph.add_many(node1, node2)

        visualizer = DAGVisualizer(graph)

        # Test that we can generate DOT through the facade
        dot_string = visualizer.to_dot(title="Component Test")

        # Verify the components are working together
        assert "digraph" in dot_string
        assert "Component Test" in dot_string
        assert "input" in dot_string
        assert "process" in dot_string

        # Test rendering with mocked subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            with tempfile.TemporaryDirectory() as tmpdir:
                result = visualizer.render_to_file(f"{tmpdir}/test", format="png")
                assert result.endswith(".png")

    @patch("hexai.agent_factory.compiler.compile_pipeline")
    @patch("pathlib.Path.exists")
    def test_with_compiled_schemas(self, mock_path, mock_compile):
        """Test loading compiled schemas from agent factory."""
        graph = DirectedGraph()
        graph._pipeline_name = "test_pipeline"

        node = NodeSpec("processor", create_test_function())
        graph.add(node)

        # Mock the compiled schema file
        mock_path.return_value = True
        mock_compile.return_value = {
            "nodes": {
                "processor": {
                    "type": "function",
                    "input_schema": {"data": "str"},
                    "output_schema": {"result": "str"},
                }
            },
            "input_schema": {"initial": "str"},
        }

        visualizer = DAGVisualizer(graph)
        dot_string = visualizer.to_dot(show_node_schemas=True)

        # The mock data should be loaded and used
        assert "processor" in dot_string

    @patch("threading.Thread")
    @patch("subprocess.run")
    def test_show_with_refactored_exporter(self, mock_run, mock_thread):
        """Test show method using the refactored FileExporter."""
        graph = DirectedGraph()
        node = NodeSpec("test", create_test_function())
        graph.add(node)

        visualizer = DAGVisualizer(graph)

        # Mock subprocess for dot command
        mock_run.return_value.returncode = 0

        # Test show doesn't raise
        visualizer.show(title="Test Show")

        # Verify subprocess was called (dot command)
        assert mock_run.called

        # Verify cleanup thread was started
        assert mock_thread.called
