"""Function-level decorators for the event system."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, TypeVar, cast

from hexai.core.registry.models import ComponentType

from .context import get_observer_manager
from .events import (
    Event,
    LifecycleEventCompleted,
    LifecycleEventFailed,
    LifecycleEventStarted,
)

EVENT_METADATA_ATTR = "__hexdag_event_metadata__"

ControlHandlerKind = Literal["control_handler"]
ObserverKind = Literal["observer"]
DecoratorKind = ControlHandlerKind | ObserverKind

TFunc = TypeVar("TFunc", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

EventType = type[Event]
EventTypesInput = EventType | Iterable[EventType] | None
ArgTransformer = Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]


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


def _normalize_component_type(value: Any) -> str | None:
    """Normalize component types from enums or strings."""
    if value is None:
        return None
    if isinstance(value, ComponentType):
        return value.value
    if isinstance(value, str):
        return value.lower()
    return None


def _extract_registry_metadata(target: Any) -> tuple[str | None, str | None, str | None]:
    """Return registry metadata (type, name, namespace) if available."""
    comp_type = _normalize_component_type(getattr(target, "_hexdag_type", None))
    comp_name = getattr(target, "_hexdag_name", None)
    namespace = getattr(target, "_hexdag_namespace", None)
    return comp_type, comp_name, namespace


def _schedule_notify(manager: Any, event: Event) -> None:
    """Schedule notification for synchronous wrappers."""
    if manager is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(manager.notify(event))
        return
    loop.create_task(manager.notify(event))


def _ensure_event_subclass(event_type: Any) -> EventType:
    """Validate and return an event subclass."""

    if not isinstance(event_type, type):
        raise TypeError(
            "event_types must contain Event subclasses; "
            f"got instance of {type(event_type).__name__}"
        )

    if not issubclass(event_type, Event):
        raise TypeError(f"event_types must contain Event subclasses; got {event_type!r}")

    return event_type


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

    raise TypeError(
        f"event_types must be a type, iterable of types, or None; got {type(event_types).__name__}"
    )


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
        raise ValueError("max_concurrency must be positive when provided")

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


def emits_events(
    *,
    component_type: str | ComponentType | None = None,
    component_name: str | None = None,
    include_args: Iterable[str] | None = None,
    include_result: bool = False,
    arg_transformer: ArgTransformer | None = None,
) -> Callable[[TFunc], TFunc]:
    """Wrap a function so that lifecycle events are emitted automatically."""

    include_args_tuple = tuple(include_args) if include_args else None

    def decorator(func: TFunc) -> TFunc:
        is_coroutine = asyncio.iscoroutinefunction(func)
        signature = inspect.signature(func)
        func_registry_type, func_registry_name, func_namespace = _extract_registry_metadata(func)
        metadata_source: Any = func

        def _infer_from_instance(instance: Any) -> tuple[str | None, str | None, str | None]:
            if instance is None:
                return None, None, None
            inst_meta = _extract_registry_metadata(instance)
            if any(inst_meta):
                return inst_meta
            class_meta = _extract_registry_metadata(getattr(instance, "__class__", object()))
            if any(class_meta):
                return class_meta
            inferred_name = getattr(instance, "name", None)
            return None, inferred_name, None

        def _heuristic_type(name: str, module: str) -> str:
            lowered = name.lower()
            if lowered == "execute":
                return "node"
            if lowered.startswith("handle"):
                return "observer"
            if lowered.startswith("evaluate"):
                return "policy"
            if "adapters" in module:
                return "adapter"
            return "function"

        def _collect_metadata(
            call_args: tuple[Any, ...], call_kwargs: dict[str, Any]
        ) -> tuple[str, str | None, dict[str, Any]]:
            inferred_type = _normalize_component_type(component_type) or func_registry_type
            inferred_name = component_name or func_registry_name
            namespace = func_namespace
            func_metadata = getattr(metadata_source, EVENT_METADATA_ATTR, None)

            instance = call_args[0] if call_args else None
            inst_type, inst_name, inst_namespace = _infer_from_instance(instance)
            if inst_type and not inferred_type:
                inferred_type = inst_type
            if inst_name and not inferred_name:
                inferred_name = inst_name
            if inst_namespace and not namespace:
                namespace = inst_namespace

            if func_metadata:
                if func_metadata.kind == "control_handler":
                    inferred_type = inferred_type or "policy"
                    inferred_name = inferred_name or func_metadata.name
                elif func_metadata.kind == "observer":
                    inferred_type = inferred_type or "observer"
                    inferred_name = inferred_name or func_metadata.id

            if not inferred_type:
                inferred_type = _heuristic_type(func.__name__, func.__module__)
            if not inferred_name:
                inferred_name = getattr(instance, "__class__", type(None)).__name__
                if inferred_name == "NoneType":
                    inferred_name = func.__name__

            metadata_payload: dict[str, Any] = {
                "module": func.__module__,
                "qualname": func.__qualname__,
            }
            if namespace:
                metadata_payload["namespace"] = namespace
            if func_metadata:
                metadata_payload["decorator"] = func_metadata.kind
                if func_metadata.kind == "control_handler":
                    metadata_payload["priority"] = func_metadata.priority
                    if func_metadata.event_types:
                        metadata_payload["event_types"] = sorted(
                            event.__name__ for event in func_metadata.event_types
                        )
                if func_metadata.kind == "observer":
                    metadata_payload["timeout"] = func_metadata.timeout
                    metadata_payload["max_concurrency"] = func_metadata.max_concurrency
            if instance is not None:
                metadata_payload["bound_instance"] = instance.__class__.__name__

            return inferred_type, inferred_name, metadata_payload

        def _collect_arguments(
            call_args: tuple[Any, ...], call_kwargs: dict[str, Any]
        ) -> dict[str, Any] | None:
            collected: dict[str, Any] = {}
            if include_args_tuple:
                try:
                    bound = signature.bind_partial(*call_args, **call_kwargs)
                    bound.apply_defaults()
                except TypeError:
                    bound = signature.bind_partial(*call_args, **call_kwargs)
                for name in include_args_tuple:
                    if name == "self":
                        continue
                    if name in bound.arguments:
                        collected[name] = bound.arguments[name]
            if arg_transformer:
                try:
                    transformed = arg_transformer(call_args, call_kwargs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Arg transformer %s raised %s; ignoring payload overrides",
                        getattr(arg_transformer, "__name__", repr(arg_transformer)),
                        exc,
                    )
                    transformed = {}
                if transformed:
                    collected.update(transformed)
            return collected or None

        async def _async_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
            manager = get_observer_manager()
            component_type_value, component_name_value, metadata_payload = _collect_metadata(
                call_args, call_kwargs
            )
            args_payload = _collect_arguments(call_args, call_kwargs)

            if manager is None:
                return await func(*call_args, **call_kwargs)

            start_event = LifecycleEventStarted(
                component_type=component_type_value,
                component_name=component_name_value,
                function_name=func.__qualname__,
                metadata=metadata_payload,
                payload=args_payload,
            )
            await manager.notify(start_event)

            start_time = time.perf_counter()
            try:
                result = await func(*call_args, **call_kwargs)
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                failed_event = LifecycleEventFailed(
                    component_type=component_type_value,
                    component_name=component_name_value,
                    function_name=func.__qualname__,
                    metadata=metadata_payload,
                    payload=args_payload,
                    duration_ms=duration_ms,
                    error=str(exc),
                    exception_type=exc.__class__.__name__,
                )
                await manager.notify(failed_event)
                raise

            duration_ms = (time.perf_counter() - start_time) * 1000
            completed_event = LifecycleEventCompleted(
                component_type=component_type_value,
                component_name=component_name_value,
                function_name=func.__qualname__,
                metadata=metadata_payload,
                payload=args_payload,
                duration_ms=duration_ms,
                result=result if include_result else None,
            )
            await manager.notify(completed_event)
            return result

        def _sync_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
            manager = get_observer_manager()
            component_type_value, component_name_value, metadata_payload = _collect_metadata(
                call_args, call_kwargs
            )
            args_payload = _collect_arguments(call_args, call_kwargs)

            if manager is not None:
                start_event = LifecycleEventStarted(
                    component_type=component_type_value,
                    component_name=component_name_value,
                    function_name=func.__qualname__,
                    metadata=metadata_payload,
                    payload=args_payload,
                )
                _schedule_notify(manager, start_event)

            start_time = time.perf_counter()
            try:
                result = func(*call_args, **call_kwargs)
            except Exception as exc:
                if manager is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    failed_event = LifecycleEventFailed(
                        component_type=component_type_value,
                        component_name=component_name_value,
                        function_name=func.__qualname__,
                        metadata=metadata_payload,
                        payload=args_payload,
                        duration_ms=duration_ms,
                        error=str(exc),
                        exception_type=exc.__class__.__name__,
                    )
                    _schedule_notify(manager, failed_event)
                raise

            if manager is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                completed_event = LifecycleEventCompleted(
                    component_type=component_type_value,
                    component_name=component_name_value,
                    function_name=func.__qualname__,
                    metadata=metadata_payload,
                    payload=args_payload,
                    duration_ms=duration_ms,
                    result=result if include_result else None,
                )
                _schedule_notify(manager, completed_event)
            return result

        wrapper = wraps(func)(_async_wrapper if is_coroutine else _sync_wrapper)
        metadata_source = wrapper
        return cast("TFunc", wrapper)

    return decorator


__all__ = [
    "EVENT_METADATA_ATTR",
    "EventDecoratorMetadata",
    "EventType",
    "EventTypesInput",
    "ArgTransformer",
    "control_handler",
    "emits_events",
    "observer",
    "normalize_event_types",
]
