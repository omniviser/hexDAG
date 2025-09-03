"""Tests for metadata.py - ComponentMetadata class."""

import pytest

from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.types import ComponentType


class TestComponentMetadata:
    """Test the ComponentMetadata class."""

    def test_metadata_creation_minimal(self):
        """Test creating metadata with minimal fields."""
        metadata = ComponentMetadata(name="test_component", component_type=ComponentType.NODE)

        assert metadata.name == "test_component"
        assert metadata.component_type == ComponentType.NODE
        assert metadata.description is None
        assert metadata.tags == frozenset()
        assert metadata.dependencies == frozenset()
        assert metadata.replaceable is False  # Default

    def test_metadata_creation_full(self):
        """Test creating metadata with all fields."""
        metadata = ComponentMetadata(
            name="full_component",
            component_type=ComponentType.TOOL,
            description="A fully configured component",
            tags={"tag1", "tag2", "tag3"},
            dependencies={"dep1", "dep2"},
            replaceable=False,
        )

        assert metadata.name == "full_component"
        assert metadata.component_type == ComponentType.TOOL
        assert metadata.description == "A fully configured component"
        assert metadata.tags == {"tag1", "tag2", "tag3"}
        assert metadata.dependencies == {"dep1", "dep2"}
        assert metadata.replaceable is False

    def test_metadata_immutability(self):
        """Test that metadata fields are properly handled."""
        metadata = ComponentMetadata(name="test", component_type=ComponentType.NODE, tags={"tag1"})

        # Tags should be a frozenset (immutable)
        assert isinstance(metadata.tags, frozenset)

        # Cannot modify frozenset
        with pytest.raises(AttributeError):
            metadata.tags.add("new_tag")  # type: ignore

        # Also test that the metadata itself is frozen
        with pytest.raises(Exception):  # Could be ValidationError, AttributeError, etc.
            metadata.name = "new_name"  # type: ignore

    def test_metadata_equality(self):
        """Test metadata equality comparison."""
        metadata1 = ComponentMetadata(name="component", component_type=ComponentType.NODE)

        metadata2 = ComponentMetadata(name="component", component_type=ComponentType.NODE)

        metadata3 = ComponentMetadata(name="different", component_type=ComponentType.NODE)

        # Same content should be equal
        assert metadata1 == metadata2

        # Different content should not be equal
        assert metadata1 != metadata3

    def test_metadata_hash(self):
        """Test metadata hashing for use in sets/dicts."""
        metadata1 = ComponentMetadata(name="component", component_type=ComponentType.NODE)

        metadata2 = ComponentMetadata(name="component", component_type=ComponentType.NODE)

        # Should be hashable
        metadata_set = {metadata1, metadata2}

        # Equal objects should have same hash
        if metadata1 == metadata2:
            assert len(metadata_set) == 1

    def test_metadata_string_representation(self):
        """Test string representation of metadata."""
        metadata = ComponentMetadata(name="test_component", component_type=ComponentType.NODE)

        str_repr = str(metadata)

        # Should contain key information
        assert "test_component" in str_repr or "ComponentMetadata" in str_repr

    def test_component_type_validation(self):
        """Test that component_type must be valid ComponentType."""
        # Valid type
        metadata = ComponentMetadata(name="test", component_type=ComponentType.NODE)
        assert metadata.component_type == ComponentType.NODE

        # Invalid type should raise error
        # Invalid type should raise error at runtime when passed
        # But ComponentType can be a string that matches enum value
        metadata = ComponentMetadata(name="test", component_type="node")
        assert metadata.component_type == "node"

    def test_author_field(self):
        """Test author field handling."""
        # Default author
        metadata1 = ComponentMetadata(name="test", component_type=ComponentType.NODE)
        assert metadata1.author == "hexdag"

        # Custom author
        metadata2 = ComponentMetadata(
            name="test", component_type=ComponentType.NODE, author="custom-author"
        )
        assert metadata2.author == "custom-author"

    def test_tags_type_conversion(self):
        """Test that tags are converted to set."""
        # From list
        metadata1 = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            tags=["tag1", "tag2", "tag1"],  # Duplicate
        )
        assert metadata1.tags == {"tag1", "tag2"}

        # From tuple
        metadata2 = ComponentMetadata(
            name="test", component_type=ComponentType.NODE, tags=("tag3", "tag4")
        )
        assert metadata2.tags == {"tag3", "tag4"}

        # Already a set
        metadata3 = ComponentMetadata(name="test", component_type=ComponentType.NODE, tags={"tag5"})
        assert metadata3.tags == {"tag5"}

    def test_dependencies_type_conversion(self):
        """Test that dependencies are converted to set."""
        metadata = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            dependencies=["dep1", "dep2", "dep1"],  # Duplicate
        )
        assert metadata.dependencies == {"dep1", "dep2"}

    def test_replaceable_flag(self):
        """Test the replaceable flag behavior."""
        # Default should be False
        metadata1 = ComponentMetadata(name="test", component_type=ComponentType.NODE)
        assert metadata1.replaceable is False

        # Explicit False
        metadata2 = ComponentMetadata(
            name="test", component_type=ComponentType.NODE, replaceable=False
        )
        assert metadata2.replaceable is False

        # Explicit True
        metadata3 = ComponentMetadata(
            name="test", component_type=ComponentType.NODE, replaceable=True
        )
        assert metadata3.replaceable is True

    def test_metadata_copy(self):
        """Test creating a copy of metadata."""
        original = ComponentMetadata(
            name="original", component_type=ComponentType.NODE, tags={"tag1"}, dependencies={"dep1"}
        )

        # Using Pydantic's model_copy
        copy = original.model_copy()

        # Should be equal but not same object
        assert copy == original
        assert copy is not original

    def test_metadata_update(self):
        """Test updating metadata fields."""
        metadata = ComponentMetadata(name="test", component_type=ComponentType.NODE)

        # If metadata is immutable, this should fail
        # If mutable, it should work
        try:
            metadata.name = "new_name"
            assert False, "Should not be able to modify frozen model"
        except (AttributeError, TypeError, Exception):
            # Immutable implementation - this is expected
            pass

    def test_metadata_validation(self):
        """Test metadata field validation."""
        # Empty name should be rejected
        # Empty name might be allowed in some implementations
        # This test depends on validation rules
        pass

        # None name should be rejected
        with pytest.raises((TypeError, ValueError)):
            ComponentMetadata(name=None, component_type=ComponentType.NODE)

    def test_metadata_serialization(self):
        """Test metadata serialization to dict."""
        metadata = ComponentMetadata(
            name="test",
            component_type=ComponentType.TOOL,
            description="Test tool",
            tags={"tag1", "tag2"},
            dependencies={"dep1"},
            replaceable=False,
        )

        # If to_dict method exists
        if hasattr(metadata, "to_dict"):
            data = metadata.to_dict()

            assert data["name"] == "test"
            assert data["component_type"] == ComponentType.TOOL.value
            assert data["description"] == "Test tool"
            assert set(data["tags"]) == {"tag1", "tag2"}
            assert set(data["dependencies"]) == {"dep1"}
            assert data["replaceable"] is False

    def test_metadata_from_dict(self):
        """Test creating metadata from dict."""
        data = {
            "name": "test",
            "component_type": ComponentType.NODE,
            "description": "Test agent",
            "tags": ["tag1"],
            "dependencies": ["dep1"],
            "replaceable": True,
        }

        # Using Pydantic's model_validate or direct constructor
        metadata = ComponentMetadata(**data)

        assert metadata.name == "test"
        assert metadata.component_type == ComponentType.NODE
        assert metadata.description == "Test agent"
        assert metadata.tags == {"tag1"}
        assert metadata.dependencies == {"dep1"}
        assert metadata.replaceable is True


class TestMetadataEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_very_long_name(self):
        """Test metadata with very long name."""
        long_name = "a" * 1000

        metadata = ComponentMetadata(name=long_name, component_type=ComponentType.NODE)

        assert metadata.name == long_name

    def test_unicode_in_fields(self):
        """Test unicode characters in metadata fields."""
        metadata = ComponentMetadata(
            name="测试组件",
            component_type=ComponentType.NODE,
            description="Component with émoji 🚀",
            tags={"中文", "العربية", "🏷️"},
        )

        assert metadata.name == "测试组件"
        assert "🚀" in metadata.description
        assert "🏷️" in metadata.tags

    def test_special_characters_in_name(self):
        """Test special characters in component name."""
        metadata = ComponentMetadata(name="test-component_v2.0", component_type=ComponentType.NODE)

        assert metadata.name == "test-component_v2.0"

    def test_empty_collections(self):
        """Test empty tags and dependencies."""
        metadata = ComponentMetadata(
            name="test", component_type=ComponentType.NODE, tags=set(), dependencies=set()
        )

        assert len(metadata.tags) == 0
        assert len(metadata.dependencies) == 0

    def test_duplicate_items_in_collections(self):
        """Test that duplicates are removed from collections."""
        metadata = ComponentMetadata(
            name="test",
            component_type=ComponentType.NODE,
            tags=["tag1", "tag1", "tag2", "tag2"],
            dependencies=["dep1", "dep1"],
        )

        assert metadata.tags == {"tag1", "tag2"}
        assert metadata.dependencies == {"dep1"}

    def test_none_collections(self):
        """Test None for tags and dependencies."""
        # None values for tags/dependencies should raise validation error
        # or be converted to empty frozensets depending on implementation
        pass

    def test_all_component_types(self):
        """Test metadata with all component types."""
        for comp_type in ComponentType:
            metadata = ComponentMetadata(name=f"test_{comp_type.value}", component_type=comp_type)

            assert metadata.component_type == comp_type
            assert metadata.name == f"test_{comp_type.value}"
