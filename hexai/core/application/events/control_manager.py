"""Control Manager: Manages execution control policies.

This is a hexagonal architecture PORT that allows policies to control
the execution flow of the pipeline through control handlers.
"""

import asyncio
import heapq
import logging
from typing import Any

from .events import Event
from .models import (
    AsyncControlHandlerFunc,
    BaseEventManager,
    ControlHandler,
    ControlHandlerFunc,
    ControlResponse,
    ControlSignal,
    ExecutionContext,
    HandlerMetadata,
)

logger = logging.getLogger(__name__)

# Priority threshold for critical handlers
CRITICAL_HANDLER_PRIORITY = 50


class ControlManager(BaseEventManager):
    """Manages control policy handlers that can affect execution.

    Key principles:
    - Handlers can control execution flow (retry, skip, fallback, fail)
    - First non-PROCEED response wins (veto pattern)
    - Handler failures are isolated and logged
    - Priority-based handler execution (lower number = higher priority)
    """

    def __init__(self) -> None:
        """Initialize the control manager."""
        super().__init__()
        # Use a list to maintain sorted order by priority
        self._priority_handlers: list[tuple[int, str, ControlHandler]] = []
        self._handler_metadata: dict[str, HandlerMetadata] = {}

    def register(self, handler: Any, **kwargs: Any) -> str:
        """Register a control handler with optional priority.

        Args
        ----
            handler: Either a ControlHandler protocol implementation or
                    a function (sync/async) that takes (event, context) -> ControlResponse
            **kwargs: Can include 'priority', 'name', 'description'

        Returns
        -------
            str: The ID/name of the registered handler
        """
        priority = kwargs.get("priority", 100)
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")

        # Generate handler ID
        if not name:
            name = getattr(handler, "__name__", f"handler_{len(self._handlers)}")

        metadata = HandlerMetadata(priority=priority, name=name, description=description)

        # Wrap function if needed
        if hasattr(handler, "handle"):
            # Already implements ControlHandler protocol
            wrapped_handler = handler
        elif callable(handler):
            # Wrap function to implement the protocol
            wrapped_handler = FunctionControlHandler(handler, metadata)
        else:
            raise TypeError(
                f"Handler must be callable or implement ControlHandler protocol, "
                f"got {type(handler)}"
            )

        # Store handler
        self._handlers[name] = wrapped_handler
        self._handler_metadata[name] = metadata

        # Add to priority queue
        heapq.heappush(self._priority_handlers, (priority, name, wrapped_handler))

        return str(name)

    def unregister(self, handler_id: str) -> bool:
        """Remove a handler by ID/name.

        Args
        ----
            handler_id: The ID/name of the handler to unregister

        Returns
        -------
            bool: True if handler was found and removed, False otherwise
        """
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            del self._handler_metadata[handler_id]

            # Rebuild priority queue without the removed handler
            self._priority_handlers = [
                (p, n, h) for p, n, h in self._priority_handlers if n != handler_id
            ]
            heapq.heapify(self._priority_handlers)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered handlers."""
        super().clear()
        self._priority_handlers.clear()
        self._handler_metadata.clear()

    async def check(self, event: Event, context: ExecutionContext) -> ControlResponse:
        """Check event against all control handlers.

        Returns
        -------
            ControlResponse with signal and optional data.
        """
        # No handlers means proceed
        if not self._priority_handlers:
            return ControlResponse()

        # Process handlers in priority order
        for priority, name, handler in sorted(self._priority_handlers):
            try:
                response = await handler.handle(event, context)

                # Validate response
                response = self._validate_response(response, name)

                # First non-PROCEED response wins
                if response.should_interrupt():
                    logger.info(
                        f"Handler {name} (priority {priority}) "
                        f"returned {response.signal.value} for {event.__class__.__name__}"
                    )
                    return response

            except Exception as e:
                logger.error(f"Control handler {name} (priority {priority}) failed: {e}")

                # For critical handlers, return ERROR signal
                if priority < CRITICAL_HANDLER_PRIORITY:
                    return ControlResponse(
                        signal=ControlSignal.ERROR,
                        data=f"Critical handler {name} failed: {e}",
                    )
                # Continue to next handler for non-critical errors
                continue

        return ControlResponse()

    def _validate_response(self, response: Any, handler_name: str) -> ControlResponse:
        """Validate and normalize handler response."""
        if response is None:
            logger.warning(f"Handler {handler_name} returned None, assuming PROCEED")
            return ControlResponse()

        if not isinstance(response, ControlResponse):
            logger.warning(
                f"Handler {handler_name} returned {type(response).__name__}, "
                f"not ControlResponse. Assuming PROCEED"
            )
            return ControlResponse()

        return response


class FunctionControlHandler:
    """Wrapper to make functions implement the ControlHandler protocol."""

    def __init__(
        self,
        func: ControlHandlerFunc | AsyncControlHandlerFunc,
        metadata: HandlerMetadata,
    ):
        self._func = func
        self.metadata = metadata
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
