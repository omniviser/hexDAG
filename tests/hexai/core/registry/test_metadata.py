"""Tests for the metadata module."""

import pytest

from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.types import ComponentType


class TestComponentMetadata:
    """Test the ComponentMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        meta = ComponentMetadata(
            name="test_component", component_type=ComponentType.NODE, namespace="test"
        )

        assert meta.name == "test_component"
        assert meta.component_type == ComponentType.NODE
        assert meta.namespace == "test"

        # Check defaults
        assert meta.description == ""
        assert meta.version == "1.0.0"
        assert meta.author == "hexdag"
        assert meta.is_core is False
        assert meta.replaceable is False
        assert meta.tags == frozenset()
        assert meta.dependencies == frozenset()

    def test_all_fields(self):
        """Test metadata with all fields specified."""
        meta = ComponentMetadata(
            name="full_component",
            component_type=ComponentType.TOOL,
            namespace="custom",
            description="A complete component",
            version="2.5.1",
            author="test_author",
            is_core=True,
            replaceable=True,
            tags=frozenset({"tag1", "tag2", "tag3"}),
            dependencies=frozenset({"dep1", "dep2"}),
        )

        assert meta.name == "full_component"
        assert meta.component_type == ComponentType.TOOL
        assert meta.namespace == "custom"
        assert meta.description == "A complete component"
        assert meta.version == "2.5.1"
        assert meta.author == "test_author"
        assert meta.is_core is True
        assert meta.replaceable is True
        assert meta.tags == frozenset({"tag1", "tag2", "tag3"})
        assert meta.dependencies == frozenset({"dep1", "dep2"})

    def test_string_component_type(self):
        """Test that string component type is converted to enum."""
        meta = ComponentMetadata(
            name="test",
            component_type="node",
            namespace="test",  # String instead of enum
        )

        # Should be converted to enum
        assert meta.component_type == ComponentType.NODE
        assert isinstance(meta.component_type, ComponentType)

    def test_tags_conversion_to_frozenset(self):
        """Test that tags are converted to frozenset."""
        # Pass regular set
        meta = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            tags={"tag1", "tag2"},  # Regular set
        )

        assert isinstance(meta.tags, frozenset)
        assert meta.tags == frozenset({"tag1", "tag2"})

        # Pass list
        meta2 = ComponentMetadata(
            name="test2",
            component_type=ComponentType.NODE,
            namespace="test",
            tags=["tag3", "tag4"],  # List
        )

        assert isinstance(meta2.tags, frozenset)
        assert meta2.tags == frozenset({"tag3", "tag4"})

    def test_dependencies_conversion_to_frozenset(self):
        """Test that dependencies are converted to frozenset."""
        # Pass regular set
        meta = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            dependencies={"dep1", "dep2"},  # Regular set
        )

        assert isinstance(meta.dependencies, frozenset)
        assert meta.dependencies == frozenset({"dep1", "dep2"})

        # Pass list
        meta2 = ComponentMetadata(
            name="test2",
            component_type=ComponentType.NODE,
            namespace="test",
            dependencies=["dep3", "dep4"],  # List
        )

        assert isinstance(meta2.dependencies, frozenset)
        assert meta2.dependencies == frozenset({"dep3", "dep4"})

    def test_runtime_fields(self):
        """Test runtime fields that are not part of __init__."""
        meta = ComponentMetadata(name="test", component_type=ComponentType.NODE, namespace="test")

        # Runtime fields should be None initially
        assert meta.component_class is None
        assert meta.config_schema is None

        # Can be set after creation
        class TestClass:
            pass

        meta.component_class = TestClass
        meta.config_schema = {"type": "object"}

        assert meta.component_class is TestClass
        assert meta.config_schema == {"type": "object"}

    def test_immutable_collections(self):
        """Test that collections are immutable (frozenset)."""
        meta = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            tags={"tag1"},
            dependencies={"dep1"},
        )

        # Should not be able to modify frozensets
        with pytest.raises(AttributeError):
            meta.tags.add("tag2")

        with pytest.raises(AttributeError):
            meta.dependencies.add("dep2")

    def test_equality(self):
        """Test metadata equality."""
        meta1 = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            description="Test component",
        )

        meta2 = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            description="Test component",
        )

        meta3 = ComponentMetadata(
            name="different",
            component_type=ComponentType.NODE,
            namespace="test",
            description="Test component",
        )

        assert meta1 == meta2
        assert meta1 != meta3

    def test_core_components(self):
        """Test metadata for core components."""
        meta = ComponentMetadata(
            name="passthrough",
            component_type=ComponentType.NODE,
            namespace="core",
            is_core=True,
            replaceable=False,
            description="Core passthrough node",
        )

        assert meta.is_core is True
        assert meta.replaceable is False
        assert meta.namespace == "core"

    def test_plugin_components(self):
        """Test metadata for plugin components."""
        meta = ComponentMetadata(
            name="custom_analyzer",
            component_type=ComponentType.NODE,
            namespace="nlp_plugin",
            is_core=False,
            replaceable=True,
            author="plugin_author",
            version="0.1.0",
            tags={"nlp", "analysis"},
            dependencies={"core:tokenizer", "core:embedder"},
        )

        assert meta.is_core is False
        assert meta.replaceable is True
        assert meta.namespace == "nlp_plugin"
        assert meta.author == "plugin_author"
        assert "nlp" in meta.tags
        assert "core:tokenizer" in meta.dependencies

    def test_invalid_component_type_string(self):
        """Test that invalid component type string raises error."""
        with pytest.raises(ValueError):
            ComponentMetadata(
                name="test",
                component_type="invalid_type",
                namespace="test",  # Invalid string
            )

    def test_empty_collections(self):
        """Test that empty collections work correctly."""
        meta = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            namespace="test",
            tags=set(),  # Empty set
            dependencies=[],  # Empty list
        )

        assert meta.tags == frozenset()
        assert meta.dependencies == frozenset()
        assert len(meta.tags) == 0
        assert len(meta.dependencies) == 0
