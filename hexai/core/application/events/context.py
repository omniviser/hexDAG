"""Execution context helpers for the event system."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

ObserverManagerVar = ContextVar[Any]  # Protocol typing occurs at runtime
_observer_manager_var: ObserverManagerVar = ContextVar("hexdag_observer_manager", default=None)


def set_observer_manager(manager: Any) -> Token[Any]:
    """Store the active ObserverManager in a context variable."""
    return _observer_manager_var.set(manager)


def reset_observer_manager(token: Token[Any]) -> None:
    """Reset the ObserverManager context variable to a previous state."""
    _observer_manager_var.reset(token)


def get_observer_manager() -> Any:
    """Retrieve the active ObserverManager for the current context."""
    return _observer_manager_var.get()


__all__ = ["get_observer_manager", "reset_observer_manager", "set_observer_manager"]
