"""Event correlation registry for suspended pipelines.

Maps external event keys to waiting pipeline runs, enabling
event-driven resume of suspended pipelines.

Usage::

    registry = EventCorrelationRegistry(storage=my_redis)

    # When pipeline suspends:
    await registry.register(WaitRegistration(
        run_id="abc-123",
        pipeline_path="negotiation.yaml",
        wait_node_name="await_reply",
        event_key="email_reply:conv-456",
    ))

    # When external event arrives:
    reg = await registry.lookup("email_reply:conv-456")
    if reg:
        result = await runner.resume_with_event(reg.pipeline_path, reg.run_id, event_data)
        await registry.remove(reg.event_key)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel.ports.data_store import SupportsKeyValue


@dataclass(frozen=True, slots=True)
class WaitRegistration:
    """Tracks a pipeline waiting for an external event.

    Attributes
    ----------
    run_id : str
        Unique identifier for the suspended pipeline run.
    pipeline_path : str
        Path to the YAML pipeline file (needed for resume).
    wait_node_name : str
        Name of the node that triggered suspension.
    event_key : str
        Correlation key for the external event.
    suspended_at : float
        Unix timestamp when the pipeline was suspended.
    timeout_at : float | None
        Unix timestamp when the wait expires.  ``None`` = no timeout.
    metadata : dict[str, Any]
        Additional metadata (e.g. on_timeout node name).
    """

    run_id: str
    pipeline_path: str
    wait_node_name: str
    event_key: str
    suspended_at: float = field(default_factory=lambda: datetime.now(UTC).timestamp())
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a storage-ready dict."""
        return {
            "run_id": self.run_id,
            "pipeline_path": self.pipeline_path,
            "wait_node_name": self.wait_node_name,
            "event_key": self.event_key,
            "suspended_at": self.suspended_at,
            "timeout_at": self.timeout_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitRegistration:
        """Reconstruct from a storage dict."""
        return cls(**data)


class EventCorrelationRegistry:
    """Storage-backed registry for suspended pipeline event correlation.

    Uses any ``SupportsKeyValue`` backend (Redis, SQLite, in-memory, etc.)
    to store wait registrations keyed by event_key.

    Parameters
    ----------
    storage : SupportsKeyValue
        Key-value storage backend.
    key_prefix : str
        Prefix for storage keys (default: ``"wait:"``).
    """

    def __init__(self, storage: SupportsKeyValue, key_prefix: str = "wait:") -> None:
        self._storage = storage
        self._prefix = key_prefix

    def _key(self, event_key: str) -> str:
        return f"{self._prefix}{event_key}"

    async def register(self, registration: WaitRegistration) -> None:
        """Register a pipeline as waiting for an external event.

        Parameters
        ----------
        registration : WaitRegistration
            The wait registration to store.
        """
        await self._storage.aset(
            self._key(registration.event_key),
            json.dumps(registration.to_dict()),
        )

    async def lookup(self, event_key: str) -> WaitRegistration | None:
        """Look up a waiting pipeline by event key.

        Parameters
        ----------
        event_key : str
            The correlation key to search for.

        Returns
        -------
        WaitRegistration | None
            The registration if found, None otherwise.
        """
        raw = await self._storage.aget(self._key(event_key))
        if raw is None:
            return None
        data = json.loads(raw) if isinstance(raw, str) else raw
        return WaitRegistration.from_dict(data)

    async def remove(self, event_key: str) -> bool:
        """Remove a wait registration after resume or cleanup.

        Parameters
        ----------
        event_key : str
            The correlation key to remove.

        Returns
        -------
        bool
            True if a registration was found and removed.
        """
        key = self._key(event_key)
        existing = await self._storage.aget(key)
        if existing is None:
            return False
        await self._storage.adelete(key)
        return True

    async def list_active(self) -> list[WaitRegistration]:
        """List all active wait registrations.

        Returns
        -------
        list[WaitRegistration]
            All currently registered waits.
        """
        keys = await self._storage.alist_keys(prefix=self._prefix)
        registrations = []
        for key in keys:
            raw = await self._storage.aget(key)
            if raw is not None:
                data = json.loads(raw) if isinstance(raw, str) else raw
                registrations.append(WaitRegistration.from_dict(data))
        return registrations

    async def get_expired(self) -> list[WaitRegistration]:
        """Get all wait registrations that have timed out.

        Returns
        -------
        list[WaitRegistration]
            Registrations where ``timeout_at`` has passed.
        """
        now = datetime.now(UTC).timestamp()
        all_active = await self.list_active()
        return [reg for reg in all_active if reg.timeout_at is not None and reg.timeout_at <= now]
