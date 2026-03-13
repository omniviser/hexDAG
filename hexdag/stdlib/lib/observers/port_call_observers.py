"""Port call observers for persisting and logging port call events.

Two complementary sinks for ``PortCallEvent`` data:

- ``PortCallStoreObserver`` — persists port calls to a ``SupportsKeyValue``
  store, queryable via VAS ``/proc/runs/<id>/port_calls/``.
- ``PortCallLogObserver`` — emits structured JSON log lines for external
  aggregators (Grafana, CloudWatch, ELK).

Both observers listen to all ``PortCallEvent`` subtypes (``LLMPortCall``,
``ToolRouterPortCall``, and any future port call events).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import Event, PortCallEvent
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.kernel.ports.tool_router import ToolRouterPortCall

if TYPE_CHECKING:
    from hexdag.kernel.ports.data_store import SupportsKeyValue

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PortCallStoreObserver — queryable port call history
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoredPortCall:
    """Serialisable record of a single port call.

    Attributes
    ----------
    port_type : str
        Port category (``"llm"``, ``"tool_router"``, etc.).
    method : str
        Method name (``"aresponse"``, ``"acall_tool"``, etc.).
    node_name : str
        DAG node that triggered the call.
    duration_ms : float
        Wall-clock duration in milliseconds.
    timestamp : float
        Unix timestamp when the event was created.
    details : dict[str, Any]
        Port-specific fields (usage, model, tool_name, etc.).
    """

    port_type: str
    method: str
    node_name: str
    duration_ms: float
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)


class PortCallStoreObserver:
    """Observer that persists port call events to a key-value store.

    Accumulates port calls in memory and provides methods to query them.
    Optionally persists to a ``SupportsKeyValue`` store for cross-run
    querying via the VAS namespace.

    Parameters
    ----------
    storage : SupportsKeyValue | None
        Optional persistent store.  If provided, ``save(run_id)`` writes
        all accumulated calls under ``port_calls:<run_id>``.

    Example
    -------
        >>> from hexdag.stdlib.lib.observers import PortCallStoreObserver
        >>> observer = PortCallStoreObserver()
        >>> # Register with observer_manager for PortCallEvent types
        >>> # ... run pipeline ...
        >>> calls = observer.get_calls()  # doctest: +SKIP
    """

    def __init__(self, storage: SupportsKeyValue | None = None) -> None:
        """Initialize with optional persistent storage backend."""
        self._storage = storage
        self._calls: list[StoredPortCall] = []

    async def handle(self, event: Event) -> None:
        """Handle port call events by storing them.

        Parameters
        ----------
        event : Event
            The event to process.  Only ``PortCallEvent`` subtypes are stored.
        """
        if not isinstance(event, PortCallEvent):
            return

        details = _extract_details(event)
        record = StoredPortCall(
            port_type=event.port_type,
            method=event.method,
            node_name=event.node_name,
            duration_ms=event.duration_ms,
            timestamp=event.timestamp.timestamp(),
            details=details,
        )
        self._calls.append(record)

    def get_calls(
        self,
        *,
        port_type: str | None = None,
        node_name: str | None = None,
        method: str | None = None,
    ) -> list[StoredPortCall]:
        """Query accumulated port calls with optional filters.

        Parameters
        ----------
        port_type : str | None
            Filter by port type (e.g. ``"llm"``, ``"tool_router"``).
        node_name : str | None
            Filter by originating node name.
        method : str | None
            Filter by method name (e.g. ``"aresponse"``).

        Returns
        -------
        list[StoredPortCall]
            Matching port call records, in chronological order.
        """
        result = self._calls
        if port_type is not None:
            result = [c for c in result if c.port_type == port_type]
        if node_name is not None:
            result = [c for c in result if c.node_name == node_name]
        if method is not None:
            result = [c for c in result if c.method == method]
        return result

    @property
    def call_count(self) -> int:
        """Total number of stored port calls."""
        return len(self._calls)

    def get_summary(self) -> dict[str, Any]:
        """Generate a summary of all port calls.

        Returns
        -------
        dict[str, Any]
            Summary with ``total_calls``, ``by_port_type``, ``by_node``,
            ``total_duration_ms``.
        """
        by_port: dict[str, int] = {}
        by_node: dict[str, int] = {}
        total_duration = 0.0

        for call in self._calls:
            by_port[call.port_type] = by_port.get(call.port_type, 0) + 1
            by_node[call.node_name] = by_node.get(call.node_name, 0) + 1
            total_duration += call.duration_ms

        return {
            "total_calls": len(self._calls),
            "by_port_type": by_port,
            "by_node": by_node,
            "total_duration_ms": total_duration,
        }

    async def save(self, run_id: str) -> int:
        """Persist all accumulated calls to storage.

        Parameters
        ----------
        run_id : str
            Unique run identifier (used as storage key).

        Returns
        -------
        int
            Number of port calls saved.

        Raises
        ------
        RuntimeError
            If no storage was configured.
        """
        if self._storage is None:
            raise RuntimeError("No storage configured. Pass storage= to PortCallStoreObserver.")
        data = [asdict(c) for c in self._calls]
        await self._storage.aset(f"port_calls:{run_id}", data)
        return len(data)

    async def load(self, run_id: str) -> list[dict[str, Any]] | None:
        """Load previously saved port calls from storage.

        Parameters
        ----------
        run_id : str
            Run identifier to load.

        Returns
        -------
        list[dict[str, Any]] | None
            List of serialised port call dicts, or None if not found.
        """
        if self._storage is None:
            return None
        result: list[dict[str, Any]] | None = await self._storage.aget(f"port_calls:{run_id}")
        return result

    def reset(self) -> None:
        """Clear all accumulated port calls."""
        self._calls.clear()


# ---------------------------------------------------------------------------
# PortCallLogObserver — structured JSON logs
# ---------------------------------------------------------------------------


class PortCallLogObserver:
    """Observer that emits structured JSON log lines for port call events.

    Designed for integration with external log aggregators (Grafana Loki,
    CloudWatch Logs, ELK stack).  Each port call produces one JSON log line
    with a consistent schema.

    Parameters
    ----------
    logger_name : str
        Logger name for output.  Defaults to the module logger.
    include_details : bool
        Whether to include port-specific details (usage, messages, etc.)
        in log output.  Defaults to True.  Set to False for minimal logs.

    Example
    -------
        >>> from hexdag.stdlib.lib.observers import PortCallLogObserver
        >>> log_observer = PortCallLogObserver()
        >>> # Register with observer_manager for PortCallEvent types
    """

    def __init__(
        self,
        logger_name: str | None = None,
        include_details: bool = True,
    ) -> None:
        """Initialize with optional logger name and detail level."""
        self._logger = get_logger(logger_name) if logger_name else logger
        self._include_details = include_details
        self._call_count = 0

    async def handle(self, event: Event) -> None:
        """Handle port call events by emitting structured JSON logs.

        Parameters
        ----------
        event : Event
            The event to process.  Only ``PortCallEvent`` subtypes are logged.
        """
        if not isinstance(event, PortCallEvent):
            return

        self._call_count += 1

        log_entry: dict[str, Any] = {
            "event": "port_call",
            "port_type": event.port_type,
            "method": event.method,
            "node_name": event.node_name,
            "duration_ms": round(event.duration_ms, 2),
            "timestamp": event.timestamp,
        }

        if self._include_details:
            log_entry["details"] = _extract_details(event)

        self._logger.info("{}", json.dumps(log_entry, default=str))

    @property
    def call_count(self) -> int:
        """Total number of port calls logged."""
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _extract_details(event: PortCallEvent) -> dict[str, Any]:
    """Extract port-specific details from a PortCallEvent subtype.

    Parameters
    ----------
    event : PortCallEvent
        The port call event.

    Returns
    -------
    dict[str, Any]
        Port-specific fields (varies by event subtype).
    """
    details: dict[str, Any] = {}

    if isinstance(event, LLMPortCall):
        if event.usage is not None:
            details["usage"] = event.usage
        if event.model is not None:
            details["model"] = event.model
        if event.tool_calls is not None:
            details["tool_calls"] = event.tool_calls
        # Omit messages and full response to avoid bloating storage/logs
        if event.response:
            details["response_length"] = len(event.response)

    elif isinstance(event, ToolRouterPortCall):
        if event.tool_name:
            details["tool_name"] = event.tool_name
        if event.params is not None:
            details["params"] = event.params
        if event.result is not None:
            # Store result type and length hint, not full result
            details["result_type"] = type(event.result).__name__
            if isinstance(event.result, dict) and "error" in event.result:
                details["error"] = event.result["error"]

    return details
