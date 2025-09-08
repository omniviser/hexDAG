"""Control Port: Manages execution control policies.

This will become a hexagonal architecture PORT that allows policies to control
the execution flow of the pipeline.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, Union

from .context import ExecutionContext
from .events import Event
from .models import ControlResponse, ControlSignal, HandlerMetadata

logger = logging.getLogger(__name__)


class ControlHandlerBase(ABC):
    """Base class for control handlers with metadata support."""

    def __init__(self, metadata: HandlerMetadata | None = None):
        self.metadata = metadata or HandlerMetadata()

    @abstractmethod
    async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Handle an event and return control response."""
        ...

    @property
    def priority(self) -> int:
        """Get handler priority."""
        return self.metadata.priority

    @property
    def name(self) -> str:
        """Get handler name."""
        return self.metadata.name or self.__class__.__name__


class ControlHandler(Protocol):
    """Protocol for control handlers that can affect execution."""

    async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Handle an event and return control response."""
        ...


# Types that can be control handlers
ControlHandlerFunc = Callable[[Any, ExecutionContext], ControlResponse]
AsyncControlHandlerFunc = Callable[[Any, ExecutionContext], Any]  # Returns awaitable
ControlHandlerLike = Union[ControlHandler, ControlHandlerFunc, AsyncControlHandlerFunc]


def _make_handler(
    func: Union[ControlHandlerFunc, AsyncControlHandlerFunc],
    metadata: HandlerMetadata | None = None,
) -> ControlHandler:
    """Convert a control handler function into a ControlHandler protocol.

    Args
    ----
        func: A function that takes (event, context) and returns ControlResponse
        metadata: Optional handler metadata with priority and description

    Returns
    -------
        An object implementing the ControlHandler protocol
    """

    class FunctionHandler(ControlHandlerBase):
        def __init__(
            self,
            fn: Union[ControlHandlerFunc, AsyncControlHandlerFunc],
            metadata: HandlerMetadata | None = None,
        ):
            super().__init__(metadata)
            self._func = fn
            self.__name__ = getattr(fn, "__name__", "anonymous_handler")
            if not self.metadata.name:
                self.metadata.name = self.__name__

        async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
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

    return FunctionHandler(func, metadata)


class EventBus:
    """Manages control handlers that can affect execution.

    Key principles:
    - Handlers can control execution flow (retry, skip, fallback, fail)
    - First non-PROCEED response wins (veto pattern)
    - Handler failures are isolated and logged
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[ControlHandler, HandlerMetadata]] = []

    def register(
        self,
        handler: ControlHandlerLike,
        priority: int = 100,
        name: str = "",
        description: str = "",
    ) -> None:
        """Register a control handler with optional priority.

        Args
        ----
            handler: Either a ControlHandler protocol implementation or
                    a function (sync/async) that takes (event, context) -> ControlResponse
            priority: Handler priority (lower = higher priority, default 100)
            name: Optional handler name for logging
            description: Optional handler description
        """
        metadata = HandlerMetadata(priority=priority, name=name, description=description)

        if isinstance(handler, ControlHandlerBase):
            # Update metadata if not already set
            if not handler.metadata.name:
                handler.metadata.name = name
            if not handler.metadata.description:
                handler.metadata.description = description
            handler.metadata.priority = priority
            self._handlers.append((handler, handler.metadata))
        elif hasattr(handler, "handle"):
            # Already implements protocol
            self._handlers.append((handler, metadata))
        elif callable(handler):
            # Wrap function to implement the protocol
            wrapped = _make_handler(handler, metadata)
            self._handlers.append((wrapped, metadata))
        else:
            raise TypeError(
                f"Handler must be callable / implement ControlHandler protocol, got {type(handler)}"
            )

        # Sort handlers by priority
        self._handlers.sort(key=lambda h: h[1].priority)

    def _validate_response(self, response: Any, handler: ControlHandler) -> ControlResponse:
        """Validate and normalize handler response."""
        handler_name = getattr(handler, "__name__", handler.__class__.__name__)

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

    async def check(self, event: Event, context: ExecutionContext) -> ControlResponse:
        """Check event against all control handlers.

        Returns
        -------
            ControlResponse with signal and optional data.
        """
        # No handlers means proceed
        if not self._handlers:
            return ControlResponse()

        for handler, metadata in self._handlers:
            try:
                response = await handler.handle(event, context)

                # Validate and normalize response
                response = self._validate_response(response, handler)

                # First non-PROCEED response wins
                if response.should_interrupt():
                    handler_name = metadata.name or getattr(
                        handler, "__name__", handler.__class__.__name__
                    )
                    logger.info(
                        f"Handler {handler_name} (priority {metadata.priority}) "
                        f"returned {response.signal.value} for {event.__class__.__name__}"
                    )
                    return response

            except Exception as e:
                handler_name = metadata.name or getattr(
                    handler, "__name__", handler.__class__.__name__
                )
                logger.error(
                    f"Control handler {handler_name} (priority {metadata.priority}) failed: {e}"
                )
                # For critical handlers, return ERROR signal
                if metadata.priority < 50:  # Consider high-priority handlers as critical
                    return ControlResponse(
                        signal=ControlSignal.ERROR,
                        data=f"Critical handler {handler_name} failed: {e}",
                    )
                # Continue to next handler for non-critical errors
                continue

        return ControlResponse()

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()
