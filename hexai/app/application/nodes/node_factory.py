"""Unified node factory for creating different types of nodes."""

from typing import Any, Callable

from ...domain.dag import NodeSpec


class NodeRegistryEntry:
    """Registry entry for a node type with its factory function and metadata."""

    def __init__(
        self,
        factory_func: Callable[..., Any],
        description: str = "",
    ):
        """Initialize the NodeRegistryEntry."""
        self.factory_func = factory_func
        self.description = description


class NodeFactory:
    """Unified factory class for creating and managing different types of nodes."""

    _registry: dict[str, NodeRegistryEntry] = {}

    @classmethod
    def register_node_type(
        cls,
        node_type: str,
        factory_func: Callable[..., Any],
        description: str = "",
    ) -> None:
        """Register a new node type with factory function.

        Args
        ----
            node_type: The type identifier for the node
            factory_func: The factory function for creating nodes
            description: Description of the node type

        Raises
        ------
            ValueError: If the node type is already registered
        """
        if node_type in cls._registry:
            raise ValueError(f"Node type '{node_type}' is already registered")

        cls._registry[node_type] = NodeRegistryEntry(factory_func, description)

    @classmethod
    def unregister_node_type(cls, node_type: str) -> None:
        """Unregister a node type.

        Args
        ----
            node_type: The type identifier for the node to unregister

        Raises
        ------
            ValueError: If the node type is not registered
        """
        if node_type not in cls._registry:
            raise ValueError(f"Node type '{node_type}' is not registered")

        del cls._registry[node_type]

    @classmethod
    def create_node(
        cls,
        node_type: str,
        node_id: str,
        **params: Any,
    ) -> NodeSpec:
        """Create a node instance using the factory.

        Args
        ----
            node_type: The type of node to create
            node_id: Unique identifier for the node
            **params: Parameters for node creation

        Returns
        -------
            A NodeSpec instance for the requested node type

        Raises
        ------
            ValueError: If the node type is not registered
            TypeError: If node creation fails
        """
        if node_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unknown node type: '{node_type}'. Available types: {available_types}"
            )

        registry_entry = cls._registry[node_type]
        try:
            node = registry_entry.factory_func(node_id, **params)
            # Ensure we return a NodeSpec
            if not isinstance(node, NodeSpec):
                raise TypeError(
                    f"Factory for '{node_type}' returned {type(node)}, expected NodeSpec"
                )
            return node
        except Exception as e:
            raise TypeError(f"Failed to create node '{node_id}' of type '{node_type}': {e}") from e

    @classmethod
    def get_node_info(cls, node_type: str) -> dict[str, Any]:
        """Get information about a registered node type.

        Args
        ----
            node_type: The node type to get info for

        Returns
        -------
            Dictionary containing node type information

        Raises
        ------
            ValueError: If the node type is not registered
        """
        if node_type not in cls._registry:
            raise ValueError(f"Unknown node type: '{node_type}'")

        registry_entry = cls._registry[node_type]
        return {
            "type": node_type,
            "description": registry_entry.description,
            "factory_function": registry_entry.factory_func.__name__,
        }

    @classmethod
    def list_registered_types(cls) -> list[str]:
        """Get a list of all registered node types.

        Returns
        -------
        List of registered node type names
        """
        return list(cls._registry.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered node types (mainly for testing)."""
        cls._registry.clear()

    @classmethod
    def get_registry_size(cls) -> int:
        """Get the number of registered node types.

        Returns
        -------
        Number of registered node types
        """
        return len(cls._registry)

    @classmethod
    def node_type_exists(cls, node_type: str) -> bool:
        """Check if a node type is registered.

        Args
        ----
            node_type: The node type to check

        Returns
        -------
            True if the node type is registered, False otherwise
        """
        return node_type in cls._registry


# Convenience functions for backward compatibility
def register_node_type(
    node_type: str, factory_func: Callable[..., Any], description: str = ""
) -> None:
    """Register a node type with the factory."""
    NodeFactory.register_node_type(node_type, factory_func, description)


def create_node(node_type: str, node_id: str, **params: Any) -> NodeSpec:
    """Create a node using the factory."""
    return NodeFactory.create_node(node_type, node_id, **params)


def list_registered_types() -> list[str]:
    """List all registered node types."""
    return NodeFactory.list_registered_types()


def clear_registry() -> None:
    """Clear the registry."""
    NodeFactory.clear_registry()


def get_registry_size() -> int:
    """Get registry size."""
    return NodeFactory.get_registry_size()
