"""Tests for component discovery module."""

from types import ModuleType

import pytest

from hexai.core.registry.decorators import node, tool
from hexai.core.registry.discovery import discover_components


class TestDiscoverComponents:
    """Test the discover_components function."""

    def test_discover_components_with_valid_module(self):
        """Test discovery with a valid module containing decorated components."""
        # Create a mock module
        module = ModuleType("test_module")
        module.__name__ = "test_module"

        # Add decorated components to the module
        @node(namespace="test")
        class TestNode:
            pass

        TestNode.__module__ = "test_module"

        @tool(namespace="test")
        def test_tool():
            pass

        test_tool.__module__ = "test_module"

        # Add components to module
        module.TestNode = TestNode
        module.test_tool = test_tool
        module.undecorated_class = type("UnDecoratedClass", (), {})

        # Discover components
        components = discover_components(module)

        # Should find only the decorated components
        assert len(components) == 2
        component_names = [name for name, _ in components]
        assert "TestNode" in component_names
        assert "test_tool" in component_names
        assert "undecorated_class" not in component_names

    def test_discover_components_with_invalid_type(self):
        """Test that discover_components raises TypeError for non-module types."""
        # Test with string
        with pytest.raises(TypeError, match="Expected ModuleType, got str"):
            discover_components("not_a_module")

        # Test with dict
        with pytest.raises(TypeError, match="Expected ModuleType, got dict"):
            discover_components({"not": "a module"})

        # Test with None
        with pytest.raises(TypeError, match="Expected ModuleType, got NoneType"):
            discover_components(None)

        # Test with class
        class NotAModule:
            pass

        with pytest.raises(TypeError, match="Expected ModuleType, got type"):
            discover_components(NotAModule)

    def test_discover_components_empty_module(self):
        """Test discovery on an empty module."""
        module = ModuleType("empty_module")
        module.__name__ = "empty_module"

        components = discover_components(module)
        assert components == []

    def test_discover_components_ignores_private_names(self):
        """Test that private names (starting with _) are ignored."""
        module = ModuleType("test_module")
        module.__name__ = "test_module"

        @node(namespace="test")
        class _PrivateNode:
            pass

        _PrivateNode.__module__ = "test_module"

        @tool(namespace="test")
        def _private_tool():
            pass

        _private_tool.__module__ = "test_module"

        module._PrivateNode = _PrivateNode
        module._private_tool = _private_tool

        components = discover_components(module)
        assert components == []

    def test_discover_components_filters_by_module(self):
        """Test that only components from the target module are discovered."""
        module = ModuleType("test_module")
        module.__name__ = "test_module"

        # Component from this module
        @node(namespace="test")
        class LocalNode:
            pass

        LocalNode.__module__ = "test_module"

        # Component from another module
        @node(namespace="test")
        class ExternalNode:
            pass

        ExternalNode.__module__ = "other_module"

        module.LocalNode = LocalNode
        module.ExternalNode = ExternalNode

        components = discover_components(module)

        # Should only find LocalNode
        assert len(components) == 1
        assert components[0][0] == "LocalNode"

    def test_discover_components_with_submodule_components(self):
        """Test that components from submodules are included."""
        module = ModuleType("parent_module")
        module.__name__ = "parent_module"

        @node(namespace="test")
        class SubmoduleNode:
            pass

        # Simulate a component from a submodule
        SubmoduleNode.__module__ = "parent_module.submodule"

        module.SubmoduleNode = SubmoduleNode

        components = discover_components(module)

        # Should include submodule component
        assert len(components) == 1
        assert components[0][0] == "SubmoduleNode"
