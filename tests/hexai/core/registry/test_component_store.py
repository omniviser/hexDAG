"""Tests for ComponentStore."""

import pytest

from hexai.core.registry.component_store import ComponentStore
from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
)
from hexai.core.registry.models import (
    ClassComponent,
    ComponentMetadata,
    ComponentType,
)


class TestComponentStore:
    """Test ComponentStore functionality."""

    @pytest.fixture
    def store(self):
        """Create a fresh component store."""
        return ComponentStore()

    def test_register_component(self, store):
        """Test registering a component."""

        class MyNode:
            pass

        metadata = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="user",
        )

        store.register(metadata, "user", is_protected=False)

        # Should be retrievable
        retrieved = store.get_metadata("my_node", "user")
        assert retrieved.name == "my_node"
        assert retrieved.namespace == "user"

    def test_register_duplicate_raises_error(self, store):
        """Test that registering duplicate component raises error."""

        class MyNode:
            pass

        metadata = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="user",
        )

        store.register(metadata, "user", is_protected=False)

        # Duplicate should raise error
        with pytest.raises(ComponentAlreadyRegisteredError):
            store.register(metadata, "user", is_protected=False)

    def test_get_metadata_by_qualified_name(self, store):
        """Test getting metadata with qualified name."""

        class MyNode:
            pass

        metadata = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="core",
        )

        store.register(metadata, "core", is_protected=True)

        # Retrieve with qualified name
        retrieved = store.get_metadata("core:my_node")
        assert retrieved.name == "my_node"
        assert retrieved.namespace == "core"

    def test_get_metadata_not_found_raises_error(self, store):
        """Test that getting non-existent component raises error."""
        with pytest.raises(ComponentNotFoundError):
            store.get_metadata("nonexistent")

    def test_search_priority(self, store):
        """Test that search respects priority order."""

        class UserNode:
            pass

        class PluginNode:
            pass

        user_meta = ComponentMetadata(
            name="shared",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=UserNode),
            namespace="user",
        )

        plugin_meta = ComponentMetadata(
            name="shared",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=PluginNode),
            namespace="plugin",
        )

        store.register(plugin_meta, "plugin", is_protected=False)
        store.register(user_meta, "user", is_protected=False)

        # Should find user first (priority: core, user, plugin)
        retrieved = store.get_metadata("shared")
        assert retrieved.namespace == "user"

    def test_list_components(self, store):
        """Test listing components with filters."""

        class MyNode:
            pass

        class MyTool:
            pass

        node_meta = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="user",
        )

        tool_meta = ComponentMetadata(
            name="my_tool",
            component_type=ComponentType.TOOL,
            component=ClassComponent(value=MyTool),
            namespace="user",
        )

        store.register(node_meta, "user", is_protected=False)
        store.register(tool_meta, "user", is_protected=False)

        # List all
        all_components = store.list_components()
        assert len(all_components) == 2

        # Filter by type
        nodes = store.list_components(component_type=ComponentType.NODE)
        assert len(nodes) == 1
        assert nodes[0].name == "my_node"

        # Filter by namespace
        user_components = store.list_components(namespace="user")
        assert len(user_components) == 2

    def test_port_names_match(self, store):
        """Test port name matching logic."""
        # Exact match
        assert store._port_names_match("llm", "llm")

        # Qualified vs unqualified
        assert store._port_names_match("core:llm", "llm")
        assert store._port_names_match("llm", "core:llm")

        # Different ports
        assert not store._port_names_match("llm", "database")

        # Different namespaces, same base name
        assert store._port_names_match("core:llm", "user:llm")

    def test_is_namespace_empty(self, store):
        """Test checking if namespace is empty."""
        assert store.is_namespace_empty("user")

        class MyNode:
            pass

        metadata = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="user",
        )

        store.register(metadata, "user", is_protected=False)

        assert not store.is_namespace_empty("user")
        assert store.is_namespace_empty("plugin")

    def test_clear(self, store):
        """Test clearing all components."""

        class MyNode:
            pass

        metadata = ComponentMetadata(
            name="my_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=MyNode),
            namespace="user",
        )

        store.register(metadata, "user", is_protected=False)
        assert len(store.list_components()) == 1

        store.clear()
        assert len(store.list_components()) == 0
