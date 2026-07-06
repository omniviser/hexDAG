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
class SupportsStateBackend(Protocol):
    """Domain database as the canonical store for entity states.

    When an ``EntityState`` implementation is given a state backend, the
    application's own storage (e.g. a ``status`` column) becomes the
    source of truth for *current state*: validation reads through
    ``aread_state`` and writes go through ``awrite_state``.  Audit history
    is NOT the backend's concern — it stays in the EntityState
    implementation (in-memory or ``SupportsCollectionStorage``).

    ``awrite_state``'s ``expected`` parameter enables compare-and-swap:
    implementations should perform a conditional write (e.g.
    ``UPDATE ... SET status = :state WHERE status = :expected``) and raise
    ``StaleStateError`` when the entity is no longer in ``expected`` —
    turning concurrent check-then-act races into explicit conflicts.
    """

    async def aread_state(self, entity_type: str, entity_id: str) -> str | None:
        """Return the current state of an entity, or None if unknown."""
        ...

    async def awrite_state(
        self,
        entity_type: str,
        entity_id: str,
        state: str,
        *,
        expected: str | None = None,
    ) -> None:
        """Write an entity's state.

        When ``expected`` is not None, the write must be conditional on the
        entity currently being in ``expected`` and raise ``StaleStateError``
        on mismatch.
        """
        ...


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

        Handlers accumulate: registering multiple handlers for the same
        entity type dispatches them in registration order. Handlers are
        called asynchronously after a successful transition. Any handler
        failure = transition failure (state rollback); side effects of
        handlers that already ran are not compensated.
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
        payload: dict[str, Any] | None = None,
        *,
        _context: TransitionContext | None = None,
    ) -> dict[str, Any]:
        """Transition an entity to a new state.

        Validates the transition against the registered state machine.
        ``payload`` carries optional domain context forwarded to transition
        handlers that declare a ``payload`` parameter.
        """
        ...
