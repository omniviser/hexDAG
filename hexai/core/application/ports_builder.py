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
        """Initialize an empty ports builder."""
        self._ports: dict[str, Any] = {}

    # Core AI Capabilities
    # --------------------

    def with_llm(self, llm: "LLM") -> Self:
        """Add a Language Model adapter.

        Args
        ----
            llm: LLM adapter instance (OpenAI, Anthropic, etc.)

        Returns
        -------
            Self for method chaining
        """
        self._ports["llm"] = llm
        return self

    def with_tool_router(self, router: "ToolRouter") -> Self:
        """Add a tool router for function calling.

        Args
        ----
            router: Tool router instance for managing tool execution

        Returns
        -------
            Self for method chaining
        """
        self._ports["tool_router"] = router
        return self

    # Storage & Persistence
    # ---------------------

    def with_database(self, database: "DatabasePort") -> Self:
        """Add a database adapter.

        Args
        ----
            database: Database adapter instance

        Returns
        -------
            Self for method chaining
        """
        self._ports["database"] = database
        return self

    def with_memory(self, memory: "Memory") -> Self:
        """Add a memory system for agents.

        Args
        ----
            memory: Memory adapter for conversation history

        Returns
        -------
            Self for method chaining
        """
        self._ports["memory"] = memory
        return self

    # Event & Control Systems
    # -----------------------

    def with_observer_manager(self, manager: "ObserverManagerPort") -> Self:
        """Add an observer manager for event monitoring.

        Args
        ----
            manager: Observer manager for read-only event handling

        Returns
        -------
            Self for method chaining
        """
        self._ports["observer_manager"] = manager
        return self

    def with_policy_manager(self, manager: "PolicyManagerPort") -> Self:
        """Add a policy manager for execution control.

        Args
        ----
            manager: Policy manager for controlling execution flow

        Returns
        -------
            Self for method chaining
        """
        self._ports["policy_manager"] = manager
        return self

    # External Integrations
    # ---------------------

    def with_api_call(self, api_call: "APICall") -> Self:
        """Add an API call adapter.

        Args
        ----
            api_call: API call adapter for external services

        Returns
        -------
            Self for method chaining
        """
        self._ports["api_call"] = api_call
        return self

    # Convenience Methods
    # ------------------

    def with_defaults(self) -> Self:
        """Add default implementations for common ports.

        This method provides sensible defaults:
        - LocalObserverManager for event observation
        - LocalPolicyManager for policy management
        - MockLLM for testing (should be overridden in production)

        Returns
        -------
            Self for method chaining
        """
        # Only add defaults if not already configured
        if "observer_manager" not in self._ports:
            try:
                from hexai.adapters.local import LocalObserverManager

                self._ports["observer_manager"] = LocalObserverManager()
            except ImportError:
                pass  # Optional dependency

        if "policy_manager" not in self._ports:
            try:
                from hexai.adapters.local import LocalPolicyManager

                self._ports["policy_manager"] = LocalPolicyManager()
            except ImportError:
                pass  # Optional dependency

        if "llm" not in self._ports:
            try:
                from hexai.adapters.mock import MockLLM

                self._ports["llm"] = MockLLM()
            except ImportError:
                pass  # Optional dependency

        return self

    def with_custom(self, key: str, port: Any) -> Self:
        """Add a custom port with any key.

        This provides backward compatibility and flexibility for
        custom port implementations.

        Args
        ----
            key: The port key name
            port: The port implementation

        Returns
        -------
            Self for method chaining
        """
        self._ports[key] = port
        return self

    def update(self, ports: dict[str, Any]) -> Self:
        """Update multiple ports at once from a dictionary.

        Useful for migrating from existing dictionary-based configs.

        Args
        ----
            ports: Dictionary of port implementations

        Returns
        -------
            Self for method chaining
        """
        self._ports.update(ports)
        return self

    # Builder Methods
    # --------------

    def build(self) -> dict[str, Any]:
        """Build and return the final ports dictionary.

        Returns
        -------
            Dictionary of configured ports for orchestrator use
        """
        return self._ports.copy()

    def clear(self) -> Self:
        """Clear all configured ports.

        Returns
        -------
            Self for method chaining
        """
        self._ports.clear()
        return self

    # Inspection Methods
    # -----------------

    def has(self, key: str) -> bool:
        """Check if a port is configured.

        Args
        ----
            key: The port key to check

        Returns
        -------
            True if the port is configured
        """
        return key in self._ports

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configured port by key.

        Args
        ----
            key: The port key
            default: Default value if not found

        Returns
        -------
            The port instance or default value
        """
        return self._ports.get(key, default)

    def keys(self) -> list[str]:
        """Get list of configured port keys.

        Returns
        -------
            List of port keys
        """
        return list(self._ports.keys())

    def __len__(self) -> int:
        """Get number of configured ports.

        Returns
        -------
            Number of configured ports
        """
        return len(self._ports)

    def __repr__(self) -> str:
        """Get string representation of builder state.

        Returns
        -------
            String showing configured ports
        """
        configured = ", ".join(self._ports.keys()) if self._ports else "none"
        return f"PortsBuilder(configured: {configured})"
