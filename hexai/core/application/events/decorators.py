"""Function-level decorators for event system metadata.

These decorators attach metadata to functions without performing any
registration side effects. Managers read the metadata at runtime to
configure control handlers and observers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

if TYPE_CHECKING:
    from .events import Event

ControlHandlerKind = Literal["control_handler"]
ObserverKind = Literal["observer"]
DecoratorKind = ControlHandlerKind | ObserverKind

TFunc = TypeVar("TFunc", bound=Callable[..., Any])


@dataclass(frozen=True)
class EventDecoratorMetadata:
    """Metadata attached to decorated event functions."""

    kind: DecoratorKind
    name: str | None = None
    priority: int | None = None
    event_types: set[type[Event]] | None = None
    description: str | None = None
    id: str | None = None
    timeout: float | None = None
    max_concurrency: int | None = None


def _normalize_event_types(event_types: Any) -> set[type[Event]] | None:
    """Normalize user-provided event types to a set of types."""
    if event_types is None:
        return None

    if isinstance(event_types, type):
        return {event_types}

    if isinstance(event_types, Iterable):
        normalized: set[type[Event]] = set()
        for event_type in event_types:
            if not isinstance(event_type, type):
                raise TypeError(
                    f"event_types must contain Event subclasses; got {type(event_type)!r}"
                )
            normalized.add(event_type)
        return normalized

    raise TypeError(
        f"event_types must be a type, iterable of types, or None; got {type(event_types)!r}"
    )


@overload
def control_handler(
    name: str,
    /,
    *,
    priority: int = 100,
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    description: str | None = None,
) -> Callable[[TFunc], TFunc]: ...


@overload
def control_handler(
    name: str,
    priority: int = 100,
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    description: str | None = None,
) -> Callable[[TFunc], TFunc]: ...


def control_handler(
    name: str,
    priority: int = 100,
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    description: str | None = None,
) -> Callable[[TFunc], TFunc]:
    """Mark a function as a control handler policy."""

    normalized_events = _normalize_event_types(event_types)

    def decorator(func: TFunc) -> TFunc:
        metadata = EventDecoratorMetadata(
            kind="control_handler",
            name=name,
            priority=priority,
            event_types=normalized_events,
            description=description,
        )
        func.__hexdag_event_metadata__ = metadata  # type: ignore[attr-defined]
        return func

    return decorator


@overload
def observer(
    *,
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    timeout: float | None = 5.0,
    max_concurrency: int | None = None,
    id: str | None = None,
) -> Callable[[TFunc], TFunc]: ...


@overload
def observer(
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    timeout: float | None = 5.0,
    max_concurrency: int | None = None,
    id: str | None = None,
) -> Callable[[TFunc], TFunc]: ...


def observer(
    event_types: Iterable[type[Event]] | type[Event] | None = None,
    timeout: float | None = 5.0,
    max_concurrency: int | None = None,
    id: str | None = None,
) -> Callable[[TFunc], TFunc]:
    """Mark a function as an observer with runtime metadata."""

    normalized_events = _normalize_event_types(event_types)

    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be positive when provided")

    def decorator(func: TFunc) -> TFunc:
        metadata = EventDecoratorMetadata(
            kind="observer",
            event_types=normalized_events,
            timeout=timeout,
            max_concurrency=max_concurrency,
            id=id,
        )
        func.__hexdag_event_metadata__ = metadata  # type: ignore[attr-defined]
        return func

    return decorator


__all__ = [
    "EventDecoratorMetadata",
    "control_handler",
    "observer",
]
