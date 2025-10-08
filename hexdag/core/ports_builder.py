"""Fluent builder for organizing orchestrator ports into logical categories.

This builder provides a clean, type-safe interface for configuring orchestrator
dependencies while maintaining backward compatibility with the flat dictionary format.

Enhanced with per-node and per-type port configuration support for fine-grained
control over port assignment across different node types and specific nodes.
"""

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from hexdag.core.ports import (
        LLM,
        APICall,
        DatabasePort,
        Memory,
        ObserverManagerPort,
        PolicyManagerPort,
        ToolRouter,
    )

from hexdag.core.orchestration.models import PortConfig, PortsConfiguration


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
        self._type_ports: dict[str, dict[str, Any]] = {}
        self._node_ports: dict[str, dict[str, Any]] = {}

    def _add_port(self, key: str, port: Any) -> Self:
        """Add a port to the internal registry.

        Args
        ----
            key: Port identifier
            port: Port implementation

        Returns
        -------
            Self for method chaining
        """
        self._ports[key] = port
        return self

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
        return self._add_port("llm", llm)

    def with_tool_router(self, router: "ToolRouter") -> Self:
        """Add a tool router for function calling.

        Args
        ----
            router: Tool router instance for managing tool execution

        Returns
        -------
            Self for method chaining
        """
        return self._add_port("tool_router", router)

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
        return self._add_port("database", database)

    def with_memory(self, memory: "Memory") -> Self:
        """Add a memory system for agents.

        Args
        ----
            memory: Memory adapter for conversation history

        Returns
        -------
            Self for method chaining
        """
        return self._add_port("memory", memory)

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
        return self._add_port("observer_manager", manager)

    def with_policy_manager(self, manager: "PolicyManagerPort") -> Self:
        """Add a policy manager for execution control.

        Args
        ----
            manager: Policy manager for controlling execution flow

        Returns
        -------
            Self for method chaining
        """
        return self._add_port("policy_manager", manager)

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
        return self._add_port("api_call", api_call)

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
                from hexdag.builtin.adapters.local import LocalObserverManager

                self._ports["observer_manager"] = LocalObserverManager()
            except ImportError:
                pass  # Optional dependency

        if "policy_manager" not in self._ports:
            try:
                from hexdag.builtin.adapters.local import LocalPolicyManager

                self._ports["policy_manager"] = LocalPolicyManager()
            except ImportError:
                pass  # Optional dependency

        if "llm" not in self._ports:
            try:
                from hexdag.builtin.adapters.mock import MockLLM

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
        return self._add_port(key, port)

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

    # Enhanced Configuration Methods
    # ------------------------------

    def for_type(self, node_type: str, **ports: Any) -> Self:
        """Configure ports for all nodes of a specific type.

        This method allows setting default ports for all nodes of a given type
        (e.g., "agent", "llm", "function"). These type-level defaults override
        global defaults but are overridden by per-node configurations.

        Args
        ----
            node_type: The node type to configure (e.g., "agent", "llm", "function")
            **ports: Port implementations as keyword arguments

        Returns
        -------
            Self for method chaining

        Examples
        --------
        Example usage::

            from hexdag.builtin.adapters.openai import OpenAIAdapter
            from hexdag.builtin.adapters.mock import MockLLM

            builder = (
                PortsBuilder()
                .with_llm(MockLLM())  # Global default
                .for_type("agent", llm=OpenAIAdapter(model="gpt-4"))  # Agent nodes
                .build_configuration()
            )

            # All agent nodes will use OpenAI, other nodes use MockLLM
            config = builder.build_configuration()
            agent_ports = config.resolve_ports("my_agent", "agent")
            assert isinstance(agent_ports["llm"].port, OpenAIAdapter)
        """
        if node_type not in self._type_ports:
            self._type_ports[node_type] = {}
        self._type_ports[node_type].update(ports)
        return self

    def for_node(self, node_name: str, **ports: Any) -> Self:
        """Configure ports for a specific node by name.

        This method allows overriding ports for individual nodes, providing
        the highest level of configuration precedence. Perfect for nodes that
        require special adapters or configurations.

        Args
        ----
            node_name: The node name to configure
            **ports: Port implementations as keyword arguments

        Returns
        -------
            Self for method chaining

        Examples
        --------
        Example usage::

            from hexdag.builtin.adapters.anthropic import AnthropicAdapter
            from hexdag.builtin.adapters.openai import OpenAIAdapter

            builder = (
                PortsBuilder()
                .for_type("agent", llm=OpenAIAdapter(model="gpt-4"))  # Agent default
                .for_node("researcher", llm=AnthropicAdapter(model="claude-3"))  # Override
                .build_configuration()
            )

            # Researcher node gets Claude, other agents get GPT-4
            config = builder.build_configuration()
            researcher_ports = config.resolve_ports("researcher", "agent")
            assert isinstance(researcher_ports["llm"].port, AnthropicAdapter)
        """
        if node_name not in self._node_ports:
            self._node_ports[node_name] = {}
        self._node_ports[node_name].update(ports)
        return self

    def build_configuration(self) -> PortsConfiguration:
        """Build a PortsConfiguration with full inheritance support.

        Creates a PortsConfiguration object that encapsulates global, per-type,
        and per-node port configurations. This provides more flexibility than
        the flat dictionary returned by build().

        Returns
        -------
        PortsConfiguration
            Configuration with port inheritance and resolution support

        Examples
        --------
        Example usage::

            from hexdag.builtin.adapters.mock import MockLLM
            from hexdag.builtin.adapters.openai import OpenAIAdapter
            from hexdag.builtin.adapters.anthropic import AnthropicAdapter

            config = (
                PortsBuilder()
                .with_llm(MockLLM())  # Global default
                .for_type("agent", llm=OpenAIAdapter(model="gpt-4"))  # Agent default
                .for_node("researcher", llm=AnthropicAdapter(model="claude-3"))  # Override
                .build_configuration()
            )

            # Resolve ports for different nodes
            researcher = config.to_flat_dict("researcher", "agent")  # AnthropicAdapter
            analyzer = config.to_flat_dict("analyzer", "agent")  # OpenAIAdapter
            transformer = config.to_flat_dict("transformer", "function")  # MockLLM

        Notes
        -----
        Resolution order: per-node > per-type > global defaults

        See Also
        --------
        build : For backward-compatible flat dictionary output
        """
        # Wrap all ports in PortConfig
        global_ports = (
            {k: PortConfig(port=v) for k, v in self._ports.items()}  # noqa: E501
            if self._ports
            else None
        )

        # Wrap type ports
        type_ports = None
        if self._type_ports:
            type_ports = {
                node_type: {k: PortConfig(port=v) for k, v in ports.items()}
                for node_type, ports in self._type_ports.items()
            }

        # Wrap node ports
        node_ports = None
        if self._node_ports:
            node_ports = {
                node_name: {k: PortConfig(port=v) for k, v in ports.items()}
                for node_name, ports in self._node_ports.items()
            }

        return PortsConfiguration(
            global_ports=global_ports,
            type_ports=type_ports,
            node_ports=node_ports,
        )
