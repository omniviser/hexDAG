"""Control Manager: Manages execution control policies.

This is a hexagonal architecture PORT that allows policies to control
the execution flow of the pipeline through control handlers.
"""

import asyncio
import heapq
import weakref
from dataclasses import dataclass
from typing import Any, cast

from .events import Event
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

# Priority threshold for critical handlers
CRITICAL_HANDLER_PRIORITY = 50


@dataclass
class HandlerEntry:
    """Entry for priority queue with all handler data consolidated."""

    priority: int
    name: str
    handler: "ControlHandler | FunctionControlHandler"
    event_types: set[type] | None
    metadata: HandlerMetadata
    deleted: bool = False

    def __lt__(self, other: "HandlerEntry") -> bool:
        """Compare by priority for heap ordering."""
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
        self, handler: ControlHandler | ControlHandlerFunc | AsyncControlHandlerFunc, **kwargs: Any
    ) -> str:
        """Register a control handler with optional priority and event filtering.

        Args
        ----
            handler: Either a ControlHandler protocol implementation or
                    a function (sync/async) that takes (event, context) -> ControlResponse
            **kwargs: Can include:
                - 'priority': Handler priority (lower = higher priority)
                - 'name': Handler name
                - 'description': Handler description
                - 'event_types': List of event types to handle (None = all events)

        Returns
        -------
            str: The ID/name of the registered handler
        """
        # Extract metadata from kwargs
        priority = kwargs.get("priority", 100)
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")
        event_types = kwargs.get("event_types")

        # Generate handler ID
        if not name:
            name = getattr(handler, "__name__", f"handler_{id(handler)}")

        # Check for duplicate
        if name in self._handler_index:
            raise ValueError(f"Handler '{name}' already registered")

        metadata = HandlerMetadata(priority=priority, name=name, description=description)

        # Wrap function if needed
        keep_alive = kwargs.get("keep_alive", False)
        wrapped_handler: ControlHandler | FunctionControlHandler
        if hasattr(handler, "handle"):
            # Already implements ControlHandler protocol
            wrapped_handler = cast("ControlHandler", handler)
        elif callable(handler):
            # Wrap function to implement the protocol
            wrapped_handler = FunctionControlHandler(handler, metadata)
            # Wrapped functions need to be kept alive
            keep_alive = True
        else:
            raise TypeError(
                f"Handler must be callable or implement ControlHandler protocol, "
                f"got {type(handler)}"
            )

        # Convert event types to set if provided
        event_filter = set(event_types) if event_types is not None else None

        # Create consolidated entry
        entry = HandlerEntry(
            priority=priority,
            name=name,
            handler=wrapped_handler,
            event_types=event_filter,
            metadata=metadata,
            deleted=False,
        )

        # Store in both structures
        heapq.heappush(self._handler_heap, entry)
        self._handler_index[name] = entry

        # Store handler with appropriate reference type
        if self._use_weak_refs and not keep_alive:
            try:
                # Try to store as weak reference in base class dict
                weak_ref = weakref.ref(wrapped_handler)
                self._handlers[name] = weak_ref
            except TypeError:
                # Can't create weak ref, use strong ref
                self._handlers[name] = wrapped_handler
                self._strong_refs[name] = wrapped_handler
        else:
            # Use strong reference
            self._handlers[name] = wrapped_handler
            if keep_alive:
                self._strong_refs[name] = wrapped_handler

        return str(name)

    def _should_handle(self, entry: HandlerEntry, event: Any) -> bool:
        """Check if handler should process this event type."""
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
            bool: True if handler was found and removed, False otherwise
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
        """Remove all registered handlers."""
        super().clear()
        self._handler_heap.clear()
        self._handler_index.clear()
        self._strong_refs.clear()
        self._deletion_count = 0

    async def check(self, event: Event, context: ExecutionContext) -> ControlResponse:
        """Check event against all control handlers.

        Only handlers registered for this event type will be consulted.

        Returns
        -------
            ControlResponse with signal and optional data.
        """
        # No handlers means proceed
        if not self._handler_heap:
            return ControlResponse()

        # Process handlers in priority order directly from heap
        # No sorting needed - heap is already ordered
        for entry in self._handler_heap:
            # Skip if handler doesn't care about this event type
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

                # For critical handlers, return ERROR signal
                if is_critical:
                    return ControlResponse(
                        signal=ControlSignal.ERROR,
                        data=f"Critical handler {entry.name} failed: {e}",
                    )
                # Continue to next handler for non-critical errors
                continue

        return ControlResponse()

    def _validate_response(self, response: Any, handler_name: str) -> ControlResponse:
        """Validate and normalize handler response."""
        if response is None:
            self._error_handler.handle_error(
                ValueError(f"Handler {handler_name} returned None, assuming PROCEED"),
                {"handler_name": handler_name, "is_critical": False},
            )
            return ControlResponse()

        if not isinstance(response, ControlResponse):
            self._error_handler.handle_error(
                TypeError(
                    f"Handler {handler_name} returned {type(response).__name__}, "
                    f"not ControlResponse. Assuming PROCEED"
                ),
                {"handler_name": handler_name, "is_critical": False},
            )
            return ControlResponse()

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
        """Handle the event by calling the wrapped function."""
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
