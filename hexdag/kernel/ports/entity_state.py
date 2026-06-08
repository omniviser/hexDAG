"""EntityState Port — contract for entity lifecycle management.

Defines the interface that any entity state implementation must satisfy.
The kernel depends only on this protocol; the concrete implementation
lives in ``hexdag.stdlib.lib.entity_state``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from hexdag.kernel.domain.entity_state import StateMachineConfig
    from hexdag.kernel.orchestration.events.events import TransitionContext


@runtime_checkable
class EntityState(Protocol):
    """Port interface for entity state management.

    Tracks entity lifecycles (e.g. order: new -> processing -> shipped)
    with validated transitions and audit history.

    Implementations must support:
    - State machine registration and validation
    - Transition handler registration
    - Async entity operations (register, transition, query)
    - Cleanup for garbage collection
    """

    # ------------------------------------------------------------------
    # Setup API (called before pipeline runs)
    # ------------------------------------------------------------------

    @abstractmethod
    def register_machine(self, config: StateMachineConfig) -> None:
        """Register a state machine config for an entity type."""
        ...

    @abstractmethod
    def register_handler(
        self,
        entity_type: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a transition handler for an entity type.

        Handlers are called asynchronously after a successful transition.
        Handler failure = transition failure (rollback).
        """
        ...

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __contains__(self, entity_type: str) -> bool:
        """Check if an entity type has a registered state machine."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over registered entity type names."""
        ...

    @abstractmethod
    def cleanup_entity(self, entity_type: str, entity_id: str) -> None:
        """Remove all in-memory data for a specific entity instance.

        Called by the lifecycle runner during garbage collection to
        prevent memory leaks after an entity reaches a terminal state.
        """
        ...

    # ------------------------------------------------------------------
    # Async entity operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def aregister_entity(
        self,
        entity_type: str,
        entity_id: str,
        initial_state: str | None = None,
    ) -> dict[str, Any]:
        """Register a new entity and set its initial state."""
        ...

    @abstractmethod
    async def aget_state(
        self,
        entity_type: str,
        entity_id: str,
    ) -> dict[str, Any] | None:
        """Get the current state of an entity."""
        ...

    @abstractmethod
    async def atransition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        reason: str | None = None,
        *,
        _context: TransitionContext | None = None,
    ) -> dict[str, Any]:
        """Transition an entity to a new state.

        Validates the transition against the registered state machine.
        """
        ...
