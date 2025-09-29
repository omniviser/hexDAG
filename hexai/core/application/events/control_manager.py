"""Control Manager: Manages execution control policies.

This is a hexagonal architecture PORT that allows policies to control
the execution flow of the pipeline through control handlers.
"""

from __future__ import annotations

import asyncio
import heapq
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, field_validator

from .decorators import (
    EVENT_METADATA_ATTR,
    EventDecoratorMetadata,
    EventType,
    EventTypesInput,
    normalize_event_types,
)
from .models import (
    AsyncControlHandlerFunc,
    BaseEventManager,
    ControlHandler,
    ControlHandlerFunc,
    ControlResponse,
    ControlSignal,
    ErrorHandler,
    EventFilterMixin,
    ExecutionContext,
    HandlerMetadata,
    LoggingErrorHandler,
)

if TYPE_CHECKING:
    from .events import Event

# Priority threshold for critical handlers
CRITICAL_HANDLER_PRIORITY = 50
DEFAULT_PRIORITY = 100


def _get_control_metadata(
    handler: Any,
) -> EventDecoratorMetadata | None:
    """Return control metadata if the handler is decorated."""
    metadata = getattr(handler, EVENT_METADATA_ATTR, None)
    if isinstance(metadata, EventDecoratorMetadata) and metadata.kind == "control_handler":
        return metadata
    return None


class ControlRegistrationConfig(BaseModel):
    """Validated configuration for control handler registration."""

    model_config = ConfigDict(extra="forbid")

    priority: int = DEFAULT_PRIORITY
    name: str | None = None
    description: str = ""
    event_types: set[EventType] | None = None
    keep_alive: bool = False

    @field_validator("event_types", mode="before")
    @classmethod
    def validate_event_types(cls, value: EventTypesInput) -> set[EventType] | None:
        return normalize_event_types(value)


def _ensure_control_response_return_type(func: Callable[..., Any]) -> None:
    """Ensure the function declares ControlResponse as its return type."""
    func_name = getattr(func, "__name__", repr(func))

    annotations = getattr(func, "__annotations__", {}) or {}
    return_type = annotations.get("return")

    if return_type is None:
        raise TypeError(f"Control handler {func_name} must declare ControlResponse as return type")

    if _is_control_response_annotation(return_type):
        return

    raise TypeError(f"Control handler {func_name} must return ControlResponse, got {return_type}")


def _is_control_response_annotation(annotation: Any) -> bool:
    """Check whether ``annotation`` represents a ControlResponse result."""
    if annotation is ControlResponse:
        return True

    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        valid = {
            ControlResponse.__name__,
            f"Awaitable[{ControlResponse.__name__}]",
            f"Coroutine[Any,Any,{ControlResponse.__name__}]",
        }
        return normalized in valid

    origin = get_origin(annotation)
    if origin is None:
        return False

    if origin is Annotated:
        args = get_args(annotation)
        return bool(args) and _is_control_response_annotation(args[0])

    if origin is Coroutine:
        args = get_args(annotation)
        if len(args) == 3:
            return _is_control_response_annotation(args[2])
        return False

    if origin in {Awaitable, asyncio.Future}:
        args = get_args(annotation)
        if not args:
            return False
        return _is_control_response_annotation(args[0])

    return False


@dataclass
class HandlerEntry:
    """Entry for priority queue with all handler data consolidated."""

    priority: int
    name: str
    handler: ControlHandler | FunctionControlHandler
    event_types: set[EventType] | None
    metadata: HandlerMetadata
    deleted: bool = False

    def __lt__(self, other: HandlerEntry) -> bool:
        """Compare by priority for heap ordering.

        Returns
        -------
        bool
            True if this entry has higher priority (lower number)
        """
        return self.priority < other.priority


class ControlManager(BaseEventManager, EventFilterMixin):
    """Manages control policy handlers that can affect execution.

    Key principles:
    - Handlers can control execution flow (retry, skip, fallback, fail)
    - First non-PROCEED response wins (veto pattern)
    - Handler failures are isolated and logged
    - Priority-based handler execution (lower number = higher priority)
    - Event type filtering for efficiency
    """

    def __init__(
        self, error_handler: ErrorHandler | None = None, use_weak_refs: bool = True
    ) -> None:
        """Initialize the control manager.

        Args
        ----
            error_handler: Optional error handler, defaults to LoggingErrorHandler
            use_weak_refs: If True, use weak references where possible to prevent memory leaks
        """
        super().__init__()
        self._error_handler = error_handler or LoggingErrorHandler()
        self._use_weak_refs = use_weak_refs
        # Single source of truth for handlers
        self._handler_heap: list[HandlerEntry] = []
        # Quick lookup by name
        self._handler_index: dict[str, HandlerEntry] = {}
        # Track deletions for periodic cleanup
        self._deletion_count = 0
        self._cleanup_threshold = 50
        # Keep wrapped functions alive
        self._strong_refs: dict[str, Any] = {}

    def register(
        self,
        handler: ControlHandler | ControlHandlerFunc | AsyncControlHandlerFunc,
        *,
        priority: int | None = None,
        name: str | None = None,
        description: str | None = None,
        event_types: EventTypesInput = None,
        keep_alive: bool = False,
        **extra: Any,
    ) -> str:
        """Register a control handler with optional priority and event filtering."""
        if extra:
            unexpected = ", ".join(sorted(extra))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        metadata = _get_control_metadata(handler)

        resolved_priority = (
            priority
            if priority is not None
            else (
                metadata.priority
                if metadata and metadata.priority is not None
                else DEFAULT_PRIORITY
            )
        )
        candidate_name = name if name is not None else (metadata.name if metadata else None)
        resolved_description = (
            description
            if description is not None
            else (metadata.description if metadata and metadata.description else "")
        )
        raw_event_types: EventTypesInput = (
            event_types if event_types is not None else (metadata.event_types if metadata else None)
        )
        normalized_event_types = (
            normalize_event_types(raw_event_types) if raw_event_types is not None else None
        )

        config = ControlRegistrationConfig(
            priority=resolved_priority,
            name=candidate_name,
            description=resolved_description,
            event_types=normalized_event_types,
            keep_alive=keep_alive,
        )

        resolved_name = config.name
        if not resolved_name:
            fallback = getattr(handler, "__name__", f"handler_{id(handler)}")
            resolved_name = fallback if isinstance(fallback, str) else str(fallback)

        if resolved_name in self._handler_index:
            raise ValueError(f"Handler '{resolved_name}' already registered")

        metadata_model = HandlerMetadata(
            priority=config.priority, name=resolved_name, description=config.description
        )

        keep_alive_flag = config.keep_alive

        if hasattr(handler, "handle"):
            wrapped_handler = cast("ControlHandler", handler)
        elif callable(handler):
            _ensure_control_response_return_type(handler)
            wrapped_handler = FunctionControlHandler(handler, metadata_model)
            keep_alive_flag = True
        else:
            raise TypeError(
                "Handler must be callable or implement ControlHandler protocol, "
                f"got {type(handler)}"
            )

        entry = HandlerEntry(
            priority=config.priority,
            name=resolved_name,
            handler=wrapped_handler,
            event_types=config.event_types,
            metadata=metadata_model,
            deleted=False,
        )

        heapq.heappush(self._handler_heap, entry)
        self._handler_index[resolved_name] = entry

        if self._use_weak_refs and not keep_alive_flag:
            try:
                weak_ref = weakref.ref(wrapped_handler)
                self._handlers[resolved_name] = weak_ref
            except TypeError:
                self._handlers[resolved_name] = wrapped_handler
                self._strong_refs[resolved_name] = wrapped_handler
        else:
            self._handlers[resolved_name] = wrapped_handler
            if keep_alive_flag:
                self._strong_refs[resolved_name] = wrapped_handler

        return resolved_name

    def _should_handle(self, entry: HandlerEntry, event: Any) -> bool:
        """Check if handler should process this event type.

        Returns
        -------
        bool
            True if handler should process this event
        """
        # Skip deleted entries
        if entry.deleted:
            return False

        return self._should_process_event(entry.event_types, event)

    def unregister(self, handler_id: str) -> bool:
        """Remove a handler by ID/name.

        Args
        ----
            handler_id: The ID/name of the handler to unregister

        Returns
        -------
        bool
            True if handler was found and removed, False otherwise
        """
        if handler_id not in self._handler_index:
            return False

        # Mark as deleted instead of rebuilding heap
        entry = self._handler_index[handler_id]
        entry.deleted = True

        # Remove from index
        del self._handler_index[handler_id]
        if handler_id in self._handlers:
            del self._handlers[handler_id]
        if handler_id in self._strong_refs:
            del self._strong_refs[handler_id]

        # Track deletions for cleanup
        self._deletion_count += 1

        # Cleanup if too many deletions accumulated
        if self._deletion_count >= self._cleanup_threshold:
            self._cleanup_heap()

        return True

    def _cleanup_heap(self) -> None:
        """Remove deleted entries from heap."""
        # Rebuild heap without deleted entries
        self._handler_heap = [e for e in self._handler_heap if not e.deleted]
        heapq.heapify(self._handler_heap)
        self._deletion_count = 0

    def clear(self) -> None:
        """Remove all registered handlers and reset internal state."""
        super().clear()
        self._handler_heap.clear()
        self._handler_index.clear()
        self._strong_refs.clear()
        self._deletion_count = 0

    async def check(self, event: Event, context: ExecutionContext) -> ControlResponse:
        """Check event against registered control handlers.

        Args
        ----
            event: Event instance to evaluate.
            context: Execution context associated with the current pipeline state.

        Handlers are evaluated in priority order until one interrupts execution.
        Only handlers whose event filters match the incoming event are invoked.

        Returns
        -------
        ControlResponse
            Response with signal and optional data
        """
        # No handlers means proceed
        if not self._handler_heap:
            return ControlResponse()

        # Process handlers in priority order directly from the heap
        # (heap is already ordered so no extra sorting is required)
        for entry in self._handler_heap:
            # Skip handlers that do not match this event type
            if not self._should_handle(entry, event):
                continue

            try:
                response = await entry.handler.handle(event, context)

                # Validate response
                response = self._validate_response(response, entry.name)

                # First non-PROCEED response wins
                if response.should_interrupt():
                    # Log via error handler for consistency
                    self._error_handler.handle_error(
                        Exception(
                            f"Handler {entry.name} (priority {entry.priority}) returned "
                            f"{response.signal.value} for {event.__class__.__name__}"
                        ),
                        {
                            "handler_name": entry.name,
                            "event_type": event.__class__.__name__,
                            "is_critical": False,
                        },
                    )
                    return response

            except Exception as e:
                is_critical = entry.priority < CRITICAL_HANDLER_PRIORITY

                self._error_handler.handle_error(
                    e,
                    {
                        "handler_name": entry.name,
                        "event_type": event.__class__.__name__,
                        "is_critical": is_critical,
                        "priority": entry.priority,
                    },
                )

                if isinstance(e, TypeError):
                    raise

                if is_critical:
                    return ControlResponse(
                        signal=ControlSignal.ERROR,
                        data=f"Critical handler {entry.name} failed: {e}",
                    )
                # Continue to next handler for non-critical errors
                continue

        return ControlResponse()

    def _validate_response(self, response: Any, handler_name: str) -> ControlResponse:
        """Validate and normalize handler response.

        Returns
        -------
        ControlResponse
            Validated and normalized control response
        """
        if response is None:
            raise TypeError(f"Handler {handler_name} returned None, ControlResponse required")

        if not isinstance(response, ControlResponse):
            raise TypeError(
                f"Handler {handler_name} returned {type(response).__name__}, not ControlResponse"
            )

        return response


class FunctionControlHandler:
    """Wrapper to make functions implement the ControlHandler protocol."""

    def __init__(
        self,
        func: ControlHandlerFunc | AsyncControlHandlerFunc,
        metadata: HandlerMetadata,
    ) -> None:
        """Initialize the function control handler."""
        self._func = func
        self._metadata = metadata
        self.__name__ = getattr(func, "__name__", "anonymous_handler")

    async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Handle the event by calling the wrapped function.

        Args
        ----
            event: The event to handle
            context: The execution context

        Returns
        -------
        ControlResponse
            Response from the wrapped handler function

        Raises
        ------
        TypeError
            If the handler must return ControlResponse, got {type(result)}
        """
        if asyncio.iscoroutinefunction(self._func):
            result = await self._func(event, context)
        else:
            result = self._func(event, context)

        # Ensure we return a ControlResponse
        if not isinstance(result, ControlResponse):
            raise TypeError(
                f"Handler {self.__name__} must return ControlResponse, got {type(result)}"
            )
        return result
