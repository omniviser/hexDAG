"""Test hexDAG ETL plugin integration."""

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.registry import registry


def test_plugin_components_registered():
    """Test that all ETL plugin components are properly registered."""

    # Test artifact storage adapter
    artifact_adapter = registry.get("local", namespace="etl")
    assert artifact_adapter is not None, "Local artifact adapter should be registered"

    # Test nodes

    # Note: Nodes are factory classes, not directly in registry like adapters
    # They're registered as node factories that create NodeSpecs
    pandas_transform_factory = registry.get("pandas_transform", namespace="etl")
    assert pandas_transform_factory is not None, "Pandas transform node should be registered"

    api_extract_factory = registry.get("api_extract", namespace="etl")
    assert api_extract_factory is not None, "API extract node should be registered"

    sql_extract_factory = registry.get("sql_extract", namespace="etl")
    assert sql_extract_factory is not None, "SQL extract node should be registered"

    sql_load_factory = registry.get("sql_load", namespace="etl")
    assert sql_load_factory is not None, "SQL load node should be registered"


def test_artifact_adapter_instantiation():
    """Test that artifact adapter can be instantiated."""
    from hexdag.core.registry import registry

    adapter_cls = registry.get("local", namespace="etl")
    assert adapter_cls is not None

    adapter = adapter_cls(base_path="/tmp/test_artifacts")
    assert adapter is not None
    assert hasattr(adapter, "write")
    assert hasattr(adapter, "read")
    assert hasattr(adapter, "list")


def test_pandas_transform_node_creation():
    """Test that pandas transform node can be created."""
    from hexdag.core.registry import registry

    factory_cls = registry.get("pandas_transform", namespace="etl")
    assert factory_cls is not None

    factory = factory_cls()

    # Create a simple node spec
    node_spec = factory(
        name="test_transform",
        operations=[{"type": "transform", "method": "pandas.DataFrame.head", "kwargs": {"n": 5}}],
    )

    assert isinstance(node_spec, NodeSpec)
    assert node_spec.name == "test_transform"
    assert "operations" in node_spec.params
