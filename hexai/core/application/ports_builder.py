"""Fluent builder for organizing orchestrator ports into logical categories.

This builder provides a clean, type-safe interface for configuring orchestrator
dependencies while maintaining backward compatibility with the flat dictionary format.
"""

from typing import Any, Self

from hexai.adapters.local import LocalObserverManager, LocalPolicyManager
from hexai.core.application.events.models import ErrorHandler
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

    def with_llm(self, adapter: LLM, key: str = "llm") -> Self:
        """Add a Language Model adapter.

        Args:
            adapter: LLM implementation (OpenAI, Anthropic, Mock, etc.)
            key: Optional custom key name (default: "llm")

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    def with_tool_router(self, router: ToolRouter, key: str = "tool_router") -> Self:
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

    def with_database(self, adapter: DatabasePort, key: str = "database") -> Self:
        """Add a database adapter.

        Args:
            adapter: Database implementation (Postgres, MongoDB, Mock, etc.)
            key: Optional custom key name (default: "database")

        Returns:
            Self for method chaining
        """
        self._ports[key] = adapter
        return self

    def with_memory(self, adapter: Memory, key: str = "memory") -> Self:
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

    def with_policy_manager(self, manager: PolicyManagerPort, key: str = "policy_manager") -> Self:
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
        self,
        manager: ObserverManagerPort | None = None,
        key: str = "observer_manager",
        *,
        max_concurrent_observers: int | None = None,
        observer_timeout: float | None = None,
        max_sync_workers: int | None = None,
        error_handler: ErrorHandler | None = None,
        use_weak_refs: bool | None = None,
    ) -> Self:
        """Add an observer manager for event monitoring.

        Args:
            manager: Observer manager instance (if None, creates LocalObserverManager with config)
            key: Optional custom key name (default: "observer_manager")
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler for observer errors
            use_weak_refs: If True, use weak references to prevent memory leaks

        Returns:
            Self for method chaining

        Example:
            ```python
            # Use provided manager
            builder.with_observer_manager(my_manager)

            # Create with custom config
            builder.with_observer_manager(
                max_concurrent_observers=20,
                observer_timeout=10.0,
                use_weak_refs=True
            )
            ```
        """
        if manager is None:
            # Create LocalObserverManager with provided config
            config_args: dict[str, Any] = {}
            if max_concurrent_observers is not None:
                config_args["max_concurrent_observers"] = max_concurrent_observers
            if observer_timeout is not None:
                config_args["observer_timeout"] = observer_timeout
            if max_sync_workers is not None:
                config_args["max_sync_workers"] = max_sync_workers
            if error_handler is not None:
                config_args["error_handler"] = error_handler
            if use_weak_refs is not None:
                config_args["use_weak_refs"] = use_weak_refs

            manager = LocalObserverManager(**config_args)

        self._ports[key] = manager
        return self

    def with_control_manager(self, manager: Any, key: str = "control_manager") -> Self:
        """DEPRECATED: Use with_policy_manager instead.

        Control manager has been replaced by policy manager.
        This method is kept for backward compatibility only.

        Args:
            manager: Control manager instance (deprecated)
            key: Optional custom key name (default: "control_manager")

        Returns:
            Self for method chaining
        """
        import warnings

        warnings.warn(
            "with_control_manager is deprecated. Use with_policy_manager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ports[key] = manager
        return self

    # External Services

    def with_api_client(self, client: APICall, key: str = "api") -> Self:
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

    # Observer Manager Factory Methods

    def with_local_observer_manager(
        self,
        max_concurrent_observers: int = 10,
        observer_timeout: float = 5.0,
        max_sync_workers: int = 4,
        error_handler: ErrorHandler | None = None,
        use_weak_refs: bool = True,
        key: str = "observer_manager",
    ) -> Self:
        """Explicitly create and configure a LocalObserverManager.

        Args:
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler for observer errors
            use_weak_refs: If True, use weak references to prevent memory leaks
            key: Optional custom key name (default: "observer_manager")

        Returns:
            Self for method chaining

        Example:
            ```python
            ports = (
                PortsBuilder()
                .with_local_observer_manager(
                    max_concurrent_observers=20,
                    observer_timeout=10.0
                )
                .build()
            )
            ```
        """
        manager = LocalObserverManager(
            max_concurrent_observers=max_concurrent_observers,
            observer_timeout=observer_timeout,
            max_sync_workers=max_sync_workers,
            error_handler=error_handler,
            use_weak_refs=use_weak_refs,
        )
        self._ports[key] = manager
        return self

    def with_observer_config(
        self,
        config: dict[str, Any],
        key: str = "observer_manager",
    ) -> Self:
        """Configure observer manager from a dictionary of settings.

        Args:
            config: Dictionary with observer manager configuration:
                - max_concurrent_observers: int
                - observer_timeout: float
                - max_sync_workers: int
                - error_handler: ErrorHandler
                - use_weak_refs: bool
            key: Optional custom key name (default: "observer_manager")

        Returns:
            Self for method chaining

        Example:
            ```python
            observer_config = {
                "max_concurrent_observers": 15,
                "observer_timeout": 8.0,
                "use_weak_refs": True
            }
            ports = (
                PortsBuilder()
                .with_observer_config(observer_config)
                .build()
            )
            ```
        """
        # Validate configuration keys
        valid_keys = {
            "max_concurrent_observers",
            "observer_timeout",
            "max_sync_workers",
            "error_handler",
            "use_weak_refs",
        }
        invalid_keys = set(config.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid observer configuration keys: {invalid_keys}")

        manager = LocalObserverManager(**config)
        self._ports[key] = manager
        return self

    # Convenience Methods

    def with_defaults(self, include_policy: bool = True) -> Self:
        """Configure default managers for common use cases.

        Sets up:
        - LocalObserverManager with default config (if not already configured)
        - LocalPolicyManager (if include_policy=True and not configured)

        This provides a working orchestrator configuration out of the box.

        Args:
            include_policy: Whether to include a default PolicyManager

        Returns:
            Self for method chaining
        """
        # Only add if not already configured
        if "observer_manager" not in self._ports:
            # Use LocalObserverManager with default safe configuration
            self._ports["observer_manager"] = LocalObserverManager(
                max_concurrent_observers=10,
                observer_timeout=5.0,
                max_sync_workers=4,
                use_weak_refs=True,
            )

        if include_policy and "policy_manager" not in self._ports:
            self._ports["policy_manager"] = LocalPolicyManager()

        return self

    def with_test_defaults(self) -> Self:
        """Configure minimal defaults suitable for testing.

        Sets up lightweight managers with minimal overhead.

        Returns:
            Self for method chaining
        """
        # Use minimal configurations for testing
        self._ports["observer_manager"] = LocalObserverManager(
            max_concurrent_observers=1,
            observer_timeout=1.0,
            max_sync_workers=1,
            use_weak_refs=False,  # Simpler for tests
        )
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
