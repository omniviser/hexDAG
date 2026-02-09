"""Test hexDAG ETL plugin integration."""

from hexdag.core.domain.dag import NodeSpec
from hexdag_plugins.hexdag_etl.hexdag_etl import (
    APIExtractNode,
    FileReaderNode,
    FileWriterNode,
    PandasTransformNode,
    SQLExtractNode,
    SQLLoadNode,
)


def test_plugin_components_importable():
    """Test that all ETL plugin components can be imported."""
    # Verify all exports are classes
    assert PandasTransformNode is not None
    assert FileReaderNode is not None
    assert FileWriterNode is not None
    assert APIExtractNode is not None
    assert SQLExtractNode is not None
    assert SQLLoadNode is not None


def test_pandas_transform_node_creation():
    """Test that pandas transform node can be created."""
    factory = PandasTransformNode()

    # Create a simple node spec
    node_spec = factory(
        name="test_transform",
        operations=[{"type": "transform", "method": "pandas.DataFrame.head", "kwargs": {"n": 5}}],
    )

    assert isinstance(node_spec, NodeSpec)
    assert node_spec.name == "test_transform"
    assert "operations" in node_spec.params


def test_file_reader_node_creation():
    """Test that file reader node can be created."""
    factory = FileReaderNode()

    node_spec = factory(
        name="test_reader",
        file_path="/tmp/test.csv",
        format="csv",
    )

    assert isinstance(node_spec, NodeSpec)
    assert node_spec.name == "test_reader"


def test_file_writer_node_creation():
    """Test that file writer node can be created."""
    factory = FileWriterNode()

    node_spec = factory(
        name="test_writer",
        file_path="/tmp/output.csv",
        format="csv",
    )

    assert isinstance(node_spec, NodeSpec)
    assert node_spec.name == "test_writer"
