"""Port interface for Long Term Memory."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hexai.core.registry.decorators import port

if TYPE_CHECKING:
    from hexai.core.ports.healthcheck import HealthStatus


@port(
    name="memory",
    namespace="core",
)
@runtime_checkable
class Memory(Protocol):
    """Protocol for long-term memory storage and retrieval.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify storage backend connectivity and availability
    """

    @abstractmethod
    async def aget(self, key: str) -> Any:
        """Retrieve a value from long-term memory asynchronously.

        Args
        ----
            key: The key to retrieve

        Returns
        -------
            The stored value, or None if key doesn't exist
        """
        pass

    @abstractmethod
    async def aset(self, key: str, value: Any) -> None:
        """Store a value in long-term memory asynchronously.

        Args
        ----
            key: The key to store under
            value: The value to store
        """
        pass

    async def ahealth_check(self) -> "HealthStatus":
        """Check memory storage backend health (optional).

        Adapters should verify:
        - Storage backend connectivity (database, file system, Redis, etc.)
        - Read/write operations
        - Storage capacity/availability

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about storage backend

        Examples
        --------
        Example usage::

            # Redis memory adapter health check
            status = await redis_memory.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.details  # {"connected_clients": 5, "used_memory_mb": 128}
        """
        ...
