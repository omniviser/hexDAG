"""Execution context for orchestrator components.

This module provides async-safe context management for cross-cutting concerns
like observer management and policy management. This eliminates parameter
drilling and provides a clean way to access these services throughout the
execution call stack.

The context is automatically propagated through async call chains, making
observer_manager, policy_manager, and other orchestration services available
to all components without explicit parameter passing.
"""

from __future__ import annotations

from contextvars import ContextVar
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.ports.observer_manager import ObserverManagerPort
    from hexai.core.ports.policy_manager import PolicyManagerPort

# Context variables for orchestrator components (async-safe)
_observer_manager_context: ContextVar[ObserverManagerPort | None] = ContextVar(
    "observer_manager", default=None
)

_policy_manager_context: ContextVar[PolicyManagerPort | None] = ContextVar(
    "policy_manager", default=None
)

_run_id_context: ContextVar[str | None] = ContextVar("run_id", default=None)

# Ports stored as immutable MappingProxyType to prevent race conditions in concurrent execution
_ports_context: ContextVar[MappingProxyType[str, Any] | None] = ContextVar("ports", default=None)


# ============================================================================
# Observer Manager Context
# ============================================================================


def set_observer_manager(manager: ObserverManagerPort | None) -> None:
    """Set observer manager for current async execution context.

    This should be called by the orchestrator at the start of DAG execution.
    All components within this context will automatically have access to
    the observer manager for event emission.

    Parameters
    ----------
    manager : ObserverManagerPort | None
        Observer manager instance, or None to clear context

    Examples
    --------
    >>> # In orchestrator
    >>> observer_manager = ports.get("observer_manager")
    >>> set_observer_manager(observer_manager)
    >>> # Now all components can emit events
    """
    _observer_manager_context.set(manager)


def get_observer_manager() -> ObserverManagerPort | None:
    """Get observer manager from current async execution context.

    This is called by components to access the observer manager for event
    emission. Returns None if not set or not in orchestrator context.

    Returns
    -------
    ObserverManagerPort | None
        Current observer manager, or None if not in orchestrator context

    Examples
    --------
    >>> # In any component
    >>> observer_manager = get_observer_manager()
    >>> if observer_manager:
    ...     await observer_manager.notify(NodeStarted(...))
    """
    return _observer_manager_context.get()


# ============================================================================
# Policy Manager Context
# ============================================================================


def set_policy_manager(manager: PolicyManagerPort | None) -> None:
    """Set policy manager for current async execution context.

    This should be called by the orchestrator at the start of DAG execution.

    Parameters
    ----------
    manager : PolicyManagerPort | None
        Policy manager instance, or None to clear context
    """
    _policy_manager_context.set(manager)


def get_policy_manager() -> PolicyManagerPort | None:
    """Get policy manager from current async execution context.

    Returns
    -------
    PolicyManagerPort | None
        Current policy manager, or None if not in orchestrator context
    """
    return _policy_manager_context.get()


# ============================================================================
# Run ID Context
# ============================================================================


def set_run_id(run_id: str | None) -> None:
    """Set run ID for current async execution context.

    Parameters
    ----------
    run_id : str | None
        Unique run identifier for this execution
    """
    _run_id_context.set(run_id)


def get_run_id() -> str | None:
    """Get run ID from current async execution context.

    Returns
    -------
    str | None
        Current run ID, or None if not in orchestrator context
    """
    return _run_id_context.get()


# ============================================================================
# Ports Context
# ============================================================================


def set_ports(ports: dict[str, Any] | None) -> None:
    """Set ports dict for current async execution context.

    This allows adapters and components to access ports without explicit passing.

    IMPORTANT: Ports are stored as immutable MappingProxyType to prevent race
    conditions when multiple nodes execute concurrently.

    Parameters
    ----------
    ports : dict[str, Any] | None
        Ports dictionary containing all available adapters and services
    """
    # Wrap in MappingProxyType to make immutable and prevent concurrent modification
    _ports_context.set(MappingProxyType(ports) if ports else None)


def get_ports() -> MappingProxyType[str, Any] | None:
    """Get immutable ports mapping from current async execution context.

    Returns
    -------
    MappingProxyType[str, Any] | None
        Immutable view of ports dictionary, or None if not in orchestrator context
    """
    return _ports_context.get()


def get_port(port_name: str) -> Any:
    """Get a specific port from current async execution context.

    Parameters
    ----------
    port_name : str
        Name of the port to retrieve (e.g., "llm", "database", "memory")

    Returns
    -------
    Any
        The port adapter, or None if not found or not in orchestrator context

    Examples
    --------
    >>> # In any component
    >>> llm = get_port("llm")
    >>> if llm:
    ...     response = await llm.aresponse(messages)
    """
    ports = _ports_context.get()
    if ports is None:
        return None
    return ports.get(port_name)


# ============================================================================
# Batch Context Management
# ============================================================================


class ExecutionContext:
    """Context manager for setting up orchestrator execution context.

    This provides a clean way to set all execution context variables at once
    using a context manager pattern.

    Examples
    --------
    >>> async with ExecutionContext(
    ...     observer_manager=observer,
    ...     policy_manager=policy,
    ...     run_id="run-123",
    ...     ports=all_ports
    ... ):
    ...     # All components can access context
    ...     result = await execute_dag(dag, inputs)
    """

    def __init__(
        self,
        observer_manager: ObserverManagerPort | None = None,
        policy_manager: PolicyManagerPort | None = None,
        run_id: str | None = None,
        ports: dict[str, Any] | None = None,
    ):
        """Initialize execution context.

        Parameters
        ----------
        observer_manager : ObserverManagerPort | None
            Observer manager for event emission
        policy_manager : PolicyManagerPort | None
            Policy manager for policy evaluation
        run_id : str | None
            Unique run identifier
        ports : dict[str, Any] | None
            Ports dictionary with all adapters
        """
        self.observer_manager = observer_manager
        self.policy_manager = policy_manager
        self.run_id = run_id
        self.ports = ports

    def __enter__(self) -> ExecutionContext:
        """Set up execution context (sync context manager)."""
        set_observer_manager(self.observer_manager)
        set_policy_manager(self.policy_manager)
        set_run_id(self.run_id)
        set_ports(self.ports)
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Clean up execution context (sync context manager)."""
        clear_execution_context()

    async def __aenter__(self) -> ExecutionContext:
        """Set up execution context (async context manager)."""
        set_observer_manager(self.observer_manager)
        set_policy_manager(self.policy_manager)
        set_run_id(self.run_id)
        set_ports(self.ports)
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Clean up execution context (async context manager)."""
        clear_execution_context()


# ============================================================================
# Cleanup
# ============================================================================


def clear_execution_context() -> None:
    """Clear all execution context variables.

    Useful for cleanup after orchestrator execution or in tests.
    """
    _observer_manager_context.set(None)
    _policy_manager_context.set(None)
    _run_id_context.set(None)
    _ports_context.set(None)
