from collections.abc import Callable
from typing import Any

from .envelope import SimpleContext, to_simple_event

Sink = Callable[[dict[str, Any]], None]


class SimpleEventEmitterObserver:
    """Observer adapter: Event → standardized envelope → sink(dict)."""

    def __init__(self, sink: Sink, context: SimpleContext):
        self._sink = sink
        self._ctx = context

    async def handle(self, event: Any) -> None:
        payload = to_simple_event(event, self._ctx)
        self._sink(payload)
