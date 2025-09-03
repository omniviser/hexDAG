"""Tests for types.py - ComponentType enum."""

from enum import StrEnum

import pytest

from hexai.core.registry.types import ComponentType


class TestComponentType:
    """Test the ComponentType enum."""

    def test_component_type_values(self):
        """Test that all component types have expected values."""
        assert ComponentType.NODE.value == "node"
        assert ComponentType.TOOL.value == "tool"
        assert ComponentType.ADAPTER.value == "adapter"
        assert ComponentType.POLICY.value == "policy"
        assert ComponentType.MEMORY.value == "memory"
        assert ComponentType.OBSERVER.value == "observer"

    def test_component_type_is_str_enum(self):
        """Test that ComponentType is a StrEnum."""
        assert issubclass(ComponentType, StrEnum)

        # String operations should work
        assert ComponentType.NODE == "node"
        assert str(ComponentType.TOOL) == "tool"
        assert ComponentType.ADAPTER.upper() == "ADAPTER"

    def test_component_type_comparison(self):
        """Test component type comparisons."""
        # Enum to enum
        assert ComponentType.NODE == ComponentType.NODE
        assert ComponentType.NODE != ComponentType.TOOL

        # Enum to string
        assert ComponentType.NODE == "node"
        assert ComponentType.TOOL == "tool"
        assert ComponentType.ADAPTER != "node"

    def test_component_type_iteration(self):
        """Test iterating over component types."""
        types = list(ComponentType)

        assert len(types) == 6  # Exactly 6 types
        assert ComponentType.NODE in types
        assert ComponentType.TOOL in types
        assert ComponentType.ADAPTER in types
        assert ComponentType.POLICY in types
        assert ComponentType.MEMORY in types
        assert ComponentType.OBSERVER in types

    def test_component_type_membership(self):
        """Test membership checks."""
        assert "node" in ComponentType._value2member_map_
        assert "tool" in ComponentType._value2member_map_
        assert "invalid" not in ComponentType._value2member_map_

    def test_component_type_from_string(self):
        """Test creating ComponentType from string."""
        assert ComponentType("node") == ComponentType.NODE
        assert ComponentType("tool") == ComponentType.TOOL
        assert ComponentType("adapter") == ComponentType.ADAPTER

        with pytest.raises(ValueError):
            ComponentType("invalid_type")

    def test_component_type_hash(self):
        """Test that ComponentType is hashable."""
        # Can be used in sets
        type_set = {ComponentType.NODE, ComponentType.TOOL, ComponentType.NODE}
        assert len(type_set) == 2

        # Can be used as dict keys
        type_dict = {ComponentType.NODE: "node_value", ComponentType.TOOL: "tool_value"}
        assert type_dict[ComponentType.NODE] == "node_value"

    def test_component_type_string_methods(self):
        """Test string methods on ComponentType."""
        # Since it's a StrEnum, string methods should work
        assert ComponentType.NODE.startswith("no")
        assert ComponentType.TOOL.endswith("ol")
        assert "apt" in ComponentType.ADAPTER
        assert ComponentType.POLICY.replace("policy", "rule") == "rule"

    def test_component_type_case_sensitive(self):
        """Test that ComponentType values are case-sensitive."""
        assert ComponentType.NODE == "node"
        assert ComponentType.NODE != "NODE"
        assert ComponentType.NODE != "Node"

        with pytest.raises(ValueError):
            ComponentType("NODE")  # Should fail - wrong case

    def test_component_type_repr(self):
        """Test string representation of ComponentType."""
        repr_str = repr(ComponentType.NODE)
        assert "ComponentType" in repr_str or "node" in repr_str

        # str should just return the value
        assert str(ComponentType.NODE) == "node"

    def test_component_type_in_collections(self):
        """Test ComponentType in various collections."""
        # List
        types_list = [ComponentType.NODE, ComponentType.TOOL]
        assert ComponentType.NODE in types_list

        # Tuple
        types_tuple = (ComponentType.ADAPTER, ComponentType.POLICY)
        assert ComponentType.ADAPTER in types_tuple

        # Set operations
        set1 = {ComponentType.NODE, ComponentType.TOOL}
        set2 = {ComponentType.TOOL, ComponentType.ADAPTER}

        intersection = set1 & set2
        assert ComponentType.TOOL in intersection
        assert len(intersection) == 1

        union = set1 | set2
        assert len(union) == 3

    def test_component_type_ordering(self):
        """Test ordering of ComponentType values."""
        # StrEnum should support ordering
        types = sorted([ComponentType.TOOL, ComponentType.NODE, ComponentType.ADAPTER])

        # Alphabetical ordering by value
        assert types[0] == ComponentType.ADAPTER  # 'adapter'
        assert types[1] == ComponentType.NODE  # 'node'
        assert types[2] == ComponentType.TOOL  # 'tool'

    def test_component_type_name_attribute(self):
        """Test the name attribute of ComponentType."""
        assert ComponentType.NODE.name == "NODE"
        assert ComponentType.TOOL.name == "TOOL"
        assert ComponentType.ADAPTER.name == "ADAPTER"

        # name is different from value
        assert ComponentType.NODE.name != ComponentType.NODE.value

    def test_component_type_all_unique(self):
        """Test that all ComponentType values are unique."""
        values = [ct.value for ct in ComponentType]
        assert len(values) == len(set(values))

        names = [ct.name for ct in ComponentType]
        assert len(names) == len(set(names))


class TestComponentTypeUsage:
    """Test practical usage patterns of ComponentType."""

    def test_function_parameter(self):
        """Test using ComponentType as function parameter."""

        def process_component(comp_type: ComponentType) -> str:
            if comp_type == ComponentType.NODE:
                return "Processing node"
            elif comp_type == ComponentType.TOOL:
                return "Processing tool"
            else:
                return f"Processing {comp_type.value}"

        assert process_component(ComponentType.NODE) == "Processing node"
        assert process_component(ComponentType.TOOL) == "Processing tool"
        assert process_component(ComponentType.ADAPTER) == "Processing adapter"

    def test_type_checking(self):
        """Test type checking with ComponentType."""

        def validate_type(comp_type: ComponentType) -> bool:
            return isinstance(comp_type, ComponentType)

        assert validate_type(ComponentType.NODE) is True
        assert validate_type(ComponentType.TOOL) is True

        # String is not ComponentType (even though it compares equal)
        assert validate_type("node") is False  # type: ignore

    def test_switch_pattern(self):
        """Test switch/match pattern with ComponentType."""

        def get_handler(comp_type: ComponentType):
            handlers = {
                ComponentType.NODE: lambda: "node_handler",
                ComponentType.TOOL: lambda: "tool_handler",
                ComponentType.ADAPTER: lambda: "adapter_handler",
            }
            return handlers.get(comp_type, lambda: "default_handler")()

        assert get_handler(ComponentType.NODE) == "node_handler"
        assert get_handler(ComponentType.TOOL) == "tool_handler"
        assert get_handler(ComponentType.MEMORY) == "default_handler"

    def test_filtering_by_type(self):
        """Test filtering components by type."""
        components = [
            ("comp1", ComponentType.NODE),
            ("comp2", ComponentType.TOOL),
            ("comp3", ComponentType.NODE),
            ("comp4", ComponentType.ADAPTER),
            ("comp5", ComponentType.TOOL),
        ]

        nodes = [c for c, t in components if t == ComponentType.NODE]
        assert len(nodes) == 2
        assert "comp1" in nodes
        assert "comp3" in nodes

        tools = [c for c, t in components if t == ComponentType.TOOL]
        assert len(tools) == 2
        assert "comp2" in tools
        assert "comp5" in tools

    def test_grouping_by_type(self):
        """Test grouping components by type."""
        from collections import defaultdict

        components = [
            ("node1", ComponentType.NODE),
            ("tool1", ComponentType.TOOL),
            ("node2", ComponentType.NODE),
            ("adapter1", ComponentType.ADAPTER),
            ("tool2", ComponentType.TOOL),
        ]

        grouped = defaultdict(list)
        for name, comp_type in components:
            grouped[comp_type].append(name)

        assert len(grouped[ComponentType.NODE]) == 2
        assert len(grouped[ComponentType.TOOL]) == 2
        assert len(grouped[ComponentType.ADAPTER]) == 1
        assert len(grouped[ComponentType.POLICY]) == 0  # No policies


class TestComponentTypeEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_type_creation(self):
        """Test creating ComponentType with invalid values."""
        with pytest.raises(ValueError):
            ComponentType("")

        with pytest.raises(ValueError):
            ComponentType("123")

        with pytest.raises(ValueError):
            ComponentType("NODE")  # Wrong case

        with pytest.raises(ValueError):
            ComponentType("nodes")  # Plural

    def test_none_handling(self):
        """Test handling None values."""
        with pytest.raises(ValueError):
            ComponentType(None)  # type: ignore

        # Comparison with None
        assert ComponentType.NODE is not None
        assert None is not ComponentType.NODE

    def test_boolean_context(self):
        """Test ComponentType in boolean context."""
        # All enum values should be truthy
        assert bool(ComponentType.NODE)
        assert bool(ComponentType.TOOL)
        assert all(ComponentType)

    def test_json_serialization(self):
        """Test JSON serialization of ComponentType."""
        import json

        # Should serialize to string value
        data = {"type": ComponentType.NODE, "name": "test"}

        # ComponentType should serialize to its value
        json_str = json.dumps(data, default=str)
        assert '"node"' in json_str or '"type": "ComponentType.NODE"' in json_str

        # Deserialization requires conversion
        loaded = json.loads(json_str)
        if loaded["type"] == "node":
            assert ComponentType(loaded["type"]) == ComponentType.NODE
