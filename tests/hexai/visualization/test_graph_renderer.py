"""Tests for GraphRenderer class."""

from unittest.mock import Mock

import pytest

from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.visualization.graph_renderer import GraphRenderer


class TestGraphRenderer:
    """Tests for GraphRenderer class."""

    @pytest.fixture
    def renderer(self):
        """Create a GraphRenderer instance."""
        return GraphRenderer()

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        graph = DirectedGraph()
        graph.add(NodeSpec(name="input", fn=lambda x: x))
        graph.add(NodeSpec(name="process", fn=lambda x: x * 2).after("input"))
        graph.add(NodeSpec(name="output", fn=lambda x: x).after("process"))
        return graph

    def test_initialization(self, renderer):
        """Test GraphRenderer initialization."""
        assert renderer.node_styles is not None
        assert "function" in renderer.node_styles
        assert "llm" in renderer.node_styles
        assert "agent" in renderer.node_styles

    def test_get_node_style(self, renderer):
        """Test node style retrieval."""
        # Test known node types
        style = renderer.get_node_style("function")
        assert style["color"] == "lightgreen"
        assert style["fillcolor"] == "lightgreen"

        style = renderer.get_node_style("llm")
        assert style["color"] == "lightblue"
        assert style["fillcolor"] == "lightblue"

        # Test unknown node type
        style = renderer.get_node_style("unknown")
        assert style["color"] == "lightgray"
        assert style["fillcolor"] == "lightgray"

        # Test None node type
        style = renderer.get_node_style(None)
        assert style["color"] == "lightgray"

    def test_get_edge_style(self, renderer):
        """Test edge style retrieval."""
        style = renderer.get_edge_style("node1", "node2")
        assert style["fontname"] == "Arial"
        assert style["fontsize"] == "8"

    def test_find_terminal_nodes(self, renderer, simple_graph):
        """Test finding terminal nodes."""
        first_nodes, last_nodes = renderer.find_terminal_nodes(simple_graph)

        assert first_nodes == ["input"]
        assert last_nodes == ["output"]

    def test_find_terminal_nodes_complex(self, renderer):
        """Test finding terminal nodes in complex graph."""
        graph = DirectedGraph()
        graph.add(NodeSpec(name="a", fn=lambda x: x))
        graph.add(NodeSpec(name="b", fn=lambda x: x))
        graph.add(NodeSpec(name="c", fn=lambda x: x).after("a", "b"))
        graph.add(NodeSpec(name="d", fn=lambda x: x).after("c"))
        graph.add(NodeSpec(name="e", fn=lambda x: x).after("c"))

        first_nodes, last_nodes = renderer.find_terminal_nodes(graph)

        assert set(first_nodes) == {"a", "b"}
        assert set(last_nodes) == {"d", "e"}

    def test_format_schema_label_none(self, renderer):
        """Test formatting schema label with None."""
        result = renderer.format_schema_label("Test Label", None)
        assert result == "Test Label"

    def test_format_schema_label_dict(self, renderer):
        """Test formatting schema label with dictionary."""
        schema = {"field1": "str", "field2": "int"}
        result = renderer.format_schema_label("Test Label", schema)

        assert "Test Label" in result
        assert "field1: str" in result
        assert "field2: int" in result

    def test_format_schema_label_dict_truncation(self, renderer):
        """Test schema label truncation for large dictionaries."""
        schema = {f"field{i}": "str" for i in range(10)}
        result = renderer.format_schema_label("Test Label", schema)

        assert "Test Label" in result
        assert "..." in result  # Should be truncated

    def test_format_schema_label_type(self, renderer):
        """Test formatting schema label with type."""
        result = renderer.format_schema_label("Test Label", str)
        assert "Test Label" in result
        assert "str" in result

    def test_format_attributes_empty(self, renderer):
        """Test formatting empty attributes."""
        result = renderer.format_attributes({})
        assert result == ""

    def test_format_attributes(self, renderer):
        """Test formatting attributes."""
        attrs = {"color": "blue", "size": 10}
        result = renderer.format_attributes(attrs)

        assert result.startswith("[")
        assert result.endswith("]")
        assert 'color="blue"' in result
        assert "size=10" in result

    def test_format_attributes_with_quotes(self, renderer):
        """Test formatting attributes with quotes."""
        attrs = {"label": 'Test "quoted" label'}
        result = renderer.format_attributes(attrs)

        assert 'label="Test \\"quoted\\" label"' in result

    def test_create_enhanced_node_label_basic(self, renderer):
        """Test creating enhanced node label."""
        node_spec = Mock()
        label = renderer.create_enhanced_node_label(
            "test_node", node_spec, None, None, "function", "my_func"
        )

        assert "test_node" in label
        assert "function" in label
        assert "my_func" in label
        assert "⚙️" in label  # Function emoji

    def test_create_enhanced_node_label_with_schemas(self, renderer):
        """Test creating enhanced node label with schemas."""
        node_spec = Mock()
        input_schema = {"input": "str", "count": "int"}
        output_schema = {"result": "str"}

        label = renderer.create_enhanced_node_label(
            "test_node", node_spec, input_schema, output_schema, "llm", None
        )

        assert "test_node" in label
        assert "🤖" in label  # LLM emoji
        assert "input: str" in label
        assert "count: int" in label
        assert "result: str" in label
        assert "⬇️ IN" in label
        assert "⬆️ OUT" in label

    def test_to_dot_basic(self, renderer, simple_graph):
        """Test basic DOT generation."""
        schemas = {}
        options = {
            "title": "Test Graph",
            "show_io_nodes": False,
            "show_intermediate_input": False,
            "show_intermediate_output": False,
        }

        dot_string = renderer.to_dot(simple_graph, schemas, options)

        assert "digraph" in dot_string
        assert "Test Graph" in dot_string
        assert "input" in dot_string
        assert "process" in dot_string
        assert "output" in dot_string
        assert "input -> process" in dot_string
        assert "process -> output" in dot_string

    def test_to_dot_with_io_nodes(self, renderer, simple_graph):
        """Test DOT generation with I/O nodes."""
        schemas = {}
        options = {
            "title": "Test Graph",
            "show_io_nodes": True,
            "input_schema": {"data": "str"},
            "output_schema": {"result": "str"},
            "show_intermediate_input": False,
            "show_intermediate_output": False,
        }

        dot_string = renderer.to_dot(simple_graph, schemas, options)

        assert "__INPUT__" in dot_string
        assert "__OUTPUT__" in dot_string
        assert "PIPELINE INPUT" in dot_string
        assert "PIPELINE OUTPUT" in dot_string
        assert "__INPUT__ -> input" in dot_string
        assert "output -> __OUTPUT__" in dot_string

    def test_to_dot_with_custom_attributes(self, renderer, simple_graph):
        """Test DOT generation with custom attributes."""
        schemas = {}
        options = {
            "title": "Test Graph",
            "show_io_nodes": False,
            "node_attributes": {"input": {"color": "red"}},
            "edge_attributes": {("input", "process"): {"style": "dashed"}},
            "show_intermediate_input": False,
            "show_intermediate_output": False,
        }

        dot_string = renderer.to_dot(simple_graph, schemas, options)

        assert "digraph" in dot_string
        # The actual attribute application would be in the DOT output

    def test_format_simple_node_label(self, renderer):
        """Test formatting simple node label."""
        node_spec = Mock()
        node_spec.type = "custom"
        node_spec.fn = Mock()
        node_spec.fn.__name__ = "my_function"

        label = renderer._format_simple_node_label("test_node", node_spec)

        assert "test_node" in label
        assert "custom" in label
        assert "my_function" in label

    def test_format_simple_node_label_no_function(self, renderer):
        """Test formatting simple node label without function."""
        node_spec = Mock()
        node_spec.type = "custom"

        # Remove fn attribute
        if hasattr(node_spec, "fn"):
            delattr(node_spec, "fn")

        label = renderer._format_simple_node_label("test_node", node_spec)

        assert "test_node" in label
        assert "custom" in label
