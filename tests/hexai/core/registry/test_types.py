"""Tests for the types module."""

import pytest

from hexai.core.registry.types import ComponentType


class TestComponentType:
    """Test the ComponentType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert ComponentType.NODE == "node"
        assert ComponentType.ADAPTER == "adapter"
        assert ComponentType.TOOL == "tool"
        assert ComponentType.POLICY == "policy"
        assert ComponentType.MEMORY == "memory"
        assert ComponentType.OBSERVER == "observer"

    def test_enum_count(self):
        """Test that we have the expected number of component types."""
        types = list(ComponentType)
        assert len(types) == 6

    def test_string_conversion(self):
        """Test conversion to string."""
        assert str(ComponentType.NODE) == "node"
        assert str(ComponentType.ADAPTER) == "adapter"
        assert str(ComponentType.TOOL) == "tool"
        assert str(ComponentType.POLICY) == "policy"
        assert str(ComponentType.MEMORY) == "memory"
        assert str(ComponentType.OBSERVER) == "observer"

    def test_from_string(self):
        """Test creating enum from string value."""
        assert ComponentType("node") == ComponentType.NODE
        assert ComponentType("adapter") == ComponentType.ADAPTER
        assert ComponentType("tool") == ComponentType.TOOL
        assert ComponentType("policy") == ComponentType.POLICY
        assert ComponentType("memory") == ComponentType.MEMORY
        assert ComponentType("observer") == ComponentType.OBSERVER

    def test_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ComponentType("invalid")

        with pytest.raises(ValueError):
            ComponentType("NODE")  # Case sensitive

        with pytest.raises(ValueError):
            ComponentType("")

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert ComponentType.NODE == ComponentType.NODE
        assert ComponentType.NODE != ComponentType.ADAPTER
        assert ComponentType.TOOL != ComponentType.POLICY

    def test_enum_in_collection(self):
        """Test using enum in collections."""
        types_set = {ComponentType.NODE, ComponentType.TOOL, ComponentType.ADAPTER}

        assert ComponentType.NODE in types_set
        assert ComponentType.TOOL in types_set
        assert ComponentType.POLICY not in types_set

        # Test in list
        types_list = [ComponentType.NODE, ComponentType.MEMORY]
        assert ComponentType.NODE in types_list
        assert ComponentType.OBSERVER not in types_list

    def test_enum_as_dict_key(self):
        """Test using enum as dictionary key."""
        type_map = {
            ComponentType.NODE: "Node handler",
            ComponentType.TOOL: "Tool handler",
            ComponentType.ADAPTER: "Adapter handler",
        }

        assert type_map[ComponentType.NODE] == "Node handler"
        assert type_map[ComponentType.TOOL] == "Tool handler"
        assert type_map.get(ComponentType.POLICY) is None

    def test_iteration(self):
        """Test iterating over enum values."""
        all_types = list(ComponentType)

        assert ComponentType.NODE in all_types
        assert ComponentType.ADAPTER in all_types
        assert ComponentType.TOOL in all_types
        assert ComponentType.POLICY in all_types
        assert ComponentType.MEMORY in all_types
        assert ComponentType.OBSERVER in all_types

        # Check order is preserved
        expected_order = [
            ComponentType.NODE,
            ComponentType.ADAPTER,
            ComponentType.TOOL,
            ComponentType.POLICY,
            ComponentType.MEMORY,
            ComponentType.OBSERVER,
        ]
        assert all_types == expected_order

    def test_enum_name_attribute(self):
        """Test that enum has name attribute."""
        assert ComponentType.NODE.name == "NODE"
        assert ComponentType.ADAPTER.name == "ADAPTER"
        assert ComponentType.TOOL.name == "TOOL"
        assert ComponentType.POLICY.name == "POLICY"
        assert ComponentType.MEMORY.name == "MEMORY"
        assert ComponentType.OBSERVER.name == "OBSERVER"

    def test_enum_value_attribute(self):
        """Test that enum has value attribute."""
        assert ComponentType.NODE.value == "node"
        assert ComponentType.ADAPTER.value == "adapter"
        assert ComponentType.TOOL.value == "tool"
        assert ComponentType.POLICY.value == "policy"
        assert ComponentType.MEMORY.value == "memory"
        assert ComponentType.OBSERVER.value == "observer"

    def test_string_enum_behavior(self):
        """Test that ComponentType behaves as StrEnum."""
        # Can be used directly as string
        result = f"Type is {ComponentType.NODE}"
        assert result == "Type is node"

        # Can be compared with strings
        assert ComponentType.NODE == "node"
        assert ComponentType.ADAPTER == "adapter"

        # Case sensitive comparison
        assert ComponentType.NODE != "NODE"
        assert ComponentType.NODE != "Node"

    def test_isinstance_checks(self):
        """Test isinstance checks with the enum."""
        assert isinstance(ComponentType.NODE, ComponentType)
        assert isinstance(ComponentType.NODE, str)  # StrEnum is also a str

        # All values should be ComponentType instances
        for comp_type in ComponentType:
            assert isinstance(comp_type, ComponentType)
            assert isinstance(comp_type, str)

    def test_repr(self):
        """Test repr of enum values."""
        # The repr should show the enum
        assert "ComponentType.NODE" in repr(ComponentType.NODE)
        assert "ComponentType.ADAPTER" in repr(ComponentType.ADAPTER)

    def test_hash(self):
        """Test that enum values are hashable."""
        # Should be able to use in set
        type_set = {ComponentType.NODE, ComponentType.TOOL, ComponentType.NODE}
        assert len(type_set) == 2  # Duplicate NODE should be removed

        # Should be able to hash
        assert hash(ComponentType.NODE) == hash(ComponentType.NODE)
        assert hash(ComponentType.NODE) != hash(ComponentType.TOOL)
