"""Function-level decorators for event system metadata.

These decorators attach metadata to functions without performing any
registration side effects. Managers read the metadata at runtime to
configure control handlers and observers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from hexdag.kernel.exceptions import TypeMismatchError, ValidationError

from .events import Event

EVENT_METADATA_ATTR = "__hexdag_event_metadata__"

ControlHandlerKind = Literal["control_handler"]
ObserverKind = Literal["observer"]
DecoratorKind = ControlHandlerKind | ObserverKind

TFunc = TypeVar("TFunc", bound=Callable[..., Any])

EventType = type[Event]
EventTypesInput = EventType | Iterable[EventType] | None


@dataclass(frozen=True)
class EventDecoratorMetadata:
    """Metadata attached to decorated event functions."""

    kind: DecoratorKind
    name: str | None = None
    priority: int | None = None
    event_types: set[EventType] | None = None
    description: str | None = None
    id: str | None = None
    timeout: float | None = None
    max_concurrency: int | None = None


def normalize_event_types(event_types: EventTypesInput) -> set[EventType] | None:
    """Normalize user-provided event types to a validated set."""

    if event_types is None:
        return None

    if isinstance(event_types, type):
        _ensure_event_subclass(event_types)
        return {event_types}

    if isinstance(event_types, Iterable):
        normalized: set[EventType] = set()
        for item in event_types:
            normalized.add(_ensure_event_subclass(item))
        return normalized

    raise TypeMismatchError(
        "event_types",
        "type, iterable of types, or None",
        type(event_types).__name__,
    )


def _ensure_event_subclass(event_type: Any) -> EventType:
    """Validate and return an event subclass."""

    if not isinstance(event_type, type):
        raise TypeMismatchError(
            "event_types",
            "Event subclass",
            f"instance of {type(event_type).__name__}",
        )

    if not issubclass(event_type, Event):
        raise TypeMismatchError("event_types", "Event subclass", repr(event_type))

    return event_type


def control_handler(
    name: str,
    *,
    priority: int = 100,
    event_types: EventTypesInput = None,
    description: str | None = None,
) -> Callable[[TFunc], TFunc]:
    """Mark a function as a control handler policy."""

    normalized_events = normalize_event_types(event_types)

    def decorator(func: TFunc) -> TFunc:
        metadata = EventDecoratorMetadata(
            kind="control_handler",
            name=name,
            priority=priority,
            event_types=normalized_events,
            description=description,
        )
        func.__dict__[EVENT_METADATA_ATTR] = metadata
        return func

    return decorator


def observer(
    *,
    event_types: EventTypesInput = None,
    timeout: float | None = 5.0,
    max_concurrency: int | None = None,
    id: str | None = None,
) -> Callable[[TFunc], TFunc]:
    """Mark a function as an observer with runtime metadata."""

    normalized_events = normalize_event_types(event_types)

    if max_concurrency is not None and max_concurrency < 1:
        raise ValidationError("max_concurrency", "must be positive", max_concurrency)

    def decorator(func: TFunc) -> TFunc:
        metadata = EventDecoratorMetadata(
            kind="observer",
            event_types=normalized_events,
            timeout=timeout,
            max_concurrency=max_concurrency,
            id=id,
        )
        func.__dict__[EVENT_METADATA_ATTR] = metadata
        return func

    return decorator


__all__ = [
    "EVENT_METADATA_ATTR",
    "EventDecoratorMetadata",
    "EventType",
    "EventTypesInput",
    "control_handler",
    "observer",
    "normalize_event_types",
]
