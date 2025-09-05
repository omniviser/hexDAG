"""Control Port: Manages execution control policies.

This will become a hexagonal architecture PORT that allows policies to control
the execution flow of the pipeline.
"""

import asyncio
import logging
from typing import Any, Callable, Protocol, Union

from .context import ExecutionContext
from .control import ControlResponse
from .events import Event

logger = logging.getLogger(__name__)


class ControlHandler(Protocol):
    """Protocol for control handlers that can affect execution."""

    async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Handle an event and return control response."""
        ...


# Types that can be control handlers
ControlHandlerFunc = Callable[[Any, ExecutionContext], ControlResponse]
AsyncControlHandlerFunc = Callable[[Any, ExecutionContext], Any]  # Returns awaitable
ControlHandlerLike = Union[ControlHandler, ControlHandlerFunc, AsyncControlHandlerFunc]


def _make_handler(func: Union[ControlHandlerFunc, AsyncControlHandlerFunc]) -> ControlHandler:
    """Convert a control handler function into a ControlHandler protocol.

    Args
    ----
        func: A function that takes (event, context) and returns ControlResponse

    Returns
    -------
        An object implementing the ControlHandler protocol
    """

    class FunctionHandler:
        def __init__(self, fn: Union[ControlHandlerFunc, AsyncControlHandlerFunc]):
            self._func = fn
            self.__name__ = getattr(fn, "__name__", "anonymous_handler")

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

    return FunctionHandler(func)


class EventBus:
    """Manages control handlers that can affect execution.

    Key principles:
    - Handlers can control execution flow (retry, skip, fallback, fail)
    - First non-PROCEED response wins (veto pattern)
    - Handler failures are isolated and logged
    """

    def __init__(self) -> None:
        self._handlers: list[ControlHandler] = []

    def register(self, handler: ControlHandlerLike) -> None:
        """Register a control handler.

        Args
        ----
            handler: Either a ControlHandler protocol implementation or
                    a function (sync/async) that takes (event, context) -> ControlResponse
        """
        if hasattr(handler, "handle"):
            # Already implements protocol
            self._handlers.append(handler)
        elif callable(handler):
            # Wrap function to implement the protocol
            self._handlers.append(_make_handler(handler))
        else:
            raise TypeError(
                f"Handler must be callable / implement ControlHandler protocol, got {type(handler)}"
            )

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

        for handler in self._handlers:
            try:
                response = await handler.handle(event, context)

                # Validate and normalize response
                response = self._validate_response(response, handler)

                # First non-PROCEED response wins
                if response.should_interrupt():
                    handler_name = getattr(handler, "__name__", handler.__class__.__name__)
                    logger.info(
                        f"Handler {handler_name} returned {response.signal.value} "
                        f"for {event.__class__.__name__}"
                    )
                    return response

            except Exception as e:
                handler_name = getattr(handler, "__name__", handler.__class__.__name__)
                logger.error(f"Control handler {handler_name} failed: {e}")
                # Continue to next handler on error
                continue

        return ControlResponse()

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()
