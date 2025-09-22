"""Fluent builder for organizing orchestrator ports into logical categories.

This builder provides a clean, type-safe interface for configuring orchestrator
dependencies while maintaining backward compatibility with the flat dictionary format.
"""

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from hexai.core.ports import (
        LLM,
        APICall,
        DatabasePort,
        Memory,
        ObserverManagerPort,
        PolicyManagerPort,
        ToolRouter,
    )


class PortsBuilder:
    """Fluent builder for organizing ports into logical categories.

    Provides a type-safe, discoverable API for configuring orchestrator
    dependencies while maintaining backward compatibility.

    Example
    -------
        ```python
        # Traditional flat dictionary approach
        ports = {
            "llm": OpenAIAdapter(),
            "database": PostgresAdapter(),
            "observer_manager": ObserverManager(),
            # ... many more mixed together
        }

        # New builder approach - organized and type-safe
        ports = (
            PortsBuilder()
            .with_llm(OpenAIAdapter())
            .with_database(PostgresAdapter())
            .with_defaults()
            .build()
        )

        # Or use with_defaults() for automatic setup
        ports = PortsBuilder().with_defaults().build()
        ```
    """

    def __init__(self) -> None:
        """Initialize empty ports builder."""
        self._ports: dict[str, Any] = {}

    # Core AI Services

    def with_llm(self, adapter: "LLM", key: str = "llm") -> Self:
        """Add a Language Model adapter.

        Args:
            adapter: LLM implementation (OpenAI, Anthropic, Mock, etc.)
            key: Optional custom key name (default: "llm")

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    def with_tool_router(self, router: "ToolRouter", key: str = "tool_router") -> Self:
        """Add a tool router for agent function calling.

        Args:
            router: Tool router implementation
            key: Optional custom key name (default: "tool_router")

        Returns:
            Self for method chaining
        """
        self._ports[key] = router
        return self

    # Data Services

    def with_database(self, adapter: "DatabasePort", key: str = "database") -> Self:
        """Add a database adapter.

        Args:
            adapter: Database implementation (Postgres, MongoDB, Mock, etc.)
            key: Optional custom key name (default: "database")

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    def with_memory(self, adapter: "Memory", key: str = "memory") -> Self:
        """Add a memory adapter for stateful operations.

        Args:
            adapter: Memory implementation (InMemory, Redis, etc.)
            key: Optional custom key name (default: "memory")

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    # Control & Monitoring

    def with_policy_manager(
        self, manager: "PolicyManagerPort", key: str = "policy_manager"
    ) -> Self:
        """Add a policy manager for execution control.

        Args:
            manager: Policy manager implementation
            key: Optional custom key name (default: "policy_manager")

        Returns:
            Self for method chaining
        """
        self._ports[key] = manager
        return self

    def with_observer_manager(
        self, manager: "ObserverManagerPort", key: str = "observer_manager"
    ) -> Self:
        """Add an observer manager for event monitoring.

        Args:
            manager: Observer manager implementation
            key: Optional custom key name (default: "observer_manager")

        Returns:
            Self for method chaining
        """
        self._ports[key] = manager
        return self

    def with_api_client(self, client: "APICall", key: str = "api") -> Self:
        """Add an API client for external service calls.

        Args:
            client: API client implementation
            key: Optional custom key name (default: "api")

        Returns:
            Self for method chaining
        """
        self._ports[key] = client
        return self

    # Custom Services

    def with_custom_port(self, key: str, adapter: Any) -> Self:
        """Add a custom port with any key and adapter.

        This method maintains full backward compatibility by allowing
        arbitrary port additions.

        Args:
            key: Port identifier
            adapter: Any adapter implementation

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    def with_custom_ports(self, ports: dict[str, Any]) -> Self:
        """Add multiple custom ports at once.

        Useful for migrating existing code or bulk additions.

        Args:
            ports: Dictionary of port key-adapter pairs

        Returns:
            Self for method chaining
        """
        self._ports.update(ports)
        return self

    # Convenience Methods

    def with_defaults(self) -> Self:
        """Placeholder for default configuration.

        Note: Default managers should be provided by the application layer,
        not imported directly in the core framework.

        Returns:
            Self for method chaining
        """
        # This method is kept for backward compatibility but does not
        # automatically create managers. Use explicit manager creation
        # in your application code.
        return self

    def with_test_defaults(self) -> Self:
        """Placeholder for test configuration.

        Note: Test managers should be provided by the test setup,
        not imported directly in the core framework.

        Returns:
            Self for method chaining
        """
        # This method is kept for backward compatibility but does not
        # automatically create managers. Use explicit manager creation
        # in your test code.
        return self

    # Builder Methods

    def build(self) -> dict[str, Any]:
        """Build the final ports dictionary.

        Returns:
            Flat dictionary compatible with current orchestrator
        """
        return self._ports.copy()

    def clear(self) -> Self:
        """Clear all configured ports.

        Returns:
            Self for method chaining
        """
        self._ports.clear()
        return self

    def has_port(self, key: str) -> bool:
        """Check if a port is configured.

        Args:
            key: Port identifier

        Returns:
            True if port is configured
        """
        return key in self._ports

    def get_port(self, key: str) -> Any | None:
        """Get a configured port by key.

        Args:
            key: Port identifier

        Returns:
            Port adapter or None if not configured
        """
        return self._ports.get(key)

    def remove_port(self, key: str) -> Self:
        """Remove a configured port.

        Args:
            key: Port identifier to remove

        Returns:
            Self for method chaining
        """
        self._ports.pop(key, None)
        return self

    @classmethod
    def from_dict(cls, ports: dict[str, Any]) -> "PortsBuilder":
        """Create a builder from an existing ports dictionary.

        Useful for migrating existing code or modifying configurations.

        Args:
            ports: Existing ports dictionary

        Returns:
            New PortsBuilder with provided ports
        """
        builder = cls()
        builder._ports = ports.copy()
        return builder

    def __repr__(self) -> str:
        """Return string representation showing configured port categories."""
        categories = {
            "AI Services": ["llm", "tool_router"],
            "Data": ["database", "memory"],
            "Control": ["policy_manager", "observer_manager", "control_manager"],
            "External": ["api"],
        }

        configured = []
        for category, keys in categories.items():
            cat_ports = [k for k in keys if k in self._ports]
            if cat_ports:
                configured.append(f"{category}: {', '.join(cat_ports)}")

        # Check for custom ports
        known_keys = set()
        for keys in categories.values():
            known_keys.update(keys)
        custom = [k for k in self._ports if k not in known_keys]
        if custom:
            configured.append(f"Custom: {', '.join(custom)}")

        return f"PortsBuilder({'; '.join(configured) if configured else 'empty'})"
