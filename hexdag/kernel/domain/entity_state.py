"""Domain models for declarative entity state machines.

Used by :class:`~hexdag.stdlib.lib.entity_state.EntityState`
to enforce state transitions on business entities (orders, tickets, etc.).

Example::

    config = StateMachineConfig(
        entity_type="order",
        states={"new", "processing", "shipped", "delivered", "cancelled"},
        initial_state="new",
        transitions={
            "new": {"processing", "cancelled"},
            "processing": {"shipped", "cancelled"},
            "shipped": {"delivered"},
        },
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hexdag.kernel.exceptions import ValidationError


@dataclass(slots=True)
class StateMachineConfig:
    """Defines the allowed states and transitions for an entity type."""

    entity_type: str
    states: set[str]
    initial_state: str
    transitions: dict[str, set[str]]

    def __post_init__(self) -> None:
        if self.initial_state not in self.states:
            raise ValidationError(
                "initial_state",
                f"not in states: {self.states}",
                self.initial_state,
            )
        for from_state, to_states in self.transitions.items():
            if from_state not in self.states:
                raise ValidationError(
                    "transitions",
                    f"source {from_state!r} not in states: {self.states}",
                    from_state,
                )
            invalid = to_states - self.states
            if invalid:
                raise ValidationError(
                    "transitions",
                    f"targets {invalid} not in states: {self.states}",
                    invalid,
                )

    def is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a transition is allowed by this config."""
        allowed = self.transitions.get(from_state, set())
        return to_state in allowed


@dataclass(slots=True)
class StateTransition:
    """Record of a single state change for an entity."""

    entity_type: str
    entity_id: str
    from_state: str | None
    to_state: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
