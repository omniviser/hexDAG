"""In-memory implementation of Memory for testing purposes."""

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


from hexdag.core.ports.memory import Memory

__all__ = ["InMemoryMemory"]


class InMemoryMemory(Memory):
    """In-memory implementation of Memory for testing.

    Features:
    - Key-value storage in memory
    - Access history tracking
    - Delay simulation
    - Size limits
    - Reset functionality
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the in-memory storage.

        Parameters
        ----------
        **kwargs : Any
            Configuration options (max_size, delay_seconds)
        """
        self.max_size = kwargs.get("max_size", 1000)
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)

        # State (not from config)
        self.storage: dict[str, Any] = {}
        self.access_history: list[dict[str, Any]] = []

    async def aget(self, key: str) -> Any:
        """Retrieve a value from memory.

        Args
        ----
            key: The key to retrieve

        Returns
        -------
            The stored value, or None if key doesn't exist
        """
        # Simulate access delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        result = self.storage.get(key)

        # Log the access
        self.access_history.append({
            "operation": "get",
            "key": key,
            "found": key in self.storage,
            "timestamp": asyncio.get_event_loop().time(),
        })

        return result

    async def aset(self, key: str, value: Any) -> None:
        """Store a value in memory.

        Args
        ----
            key: The key to store under
            value: The value to store

        Raises
        ------
        MemoryError
            If max_size is exceeded
        """
        # Simulate access delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Check size limit
        if (
            self.max_size is not None
            and key not in self.storage
            and len(self.storage) >= self.max_size
        ):
            raise MemoryError(f"Memory limit of {self.max_size} items exceeded")

        self.storage[key] = value

        # Log the access
        self.access_history.append({
            "operation": "set",
            "key": key,
            "value_type": type(value).__name__,
            "timestamp": asyncio.get_event_loop().time(),
        })

    def clear(self) -> None:
        """Clear all stored data."""
        self.storage.clear()

    def reset(self) -> None:
        """Reset the memory to initial state, clearing both data and history."""
        self.storage.clear()
        self.access_history.clear()

    def get_access_history(self) -> list[dict[str, Any]]:
        """Get the history of all memory access operations.

        Returns
        -------
        list[dict[str, Any]]
            List of access history records with operation, key, and timestamp
        """
        return self.access_history.copy()

    def get_stored_keys(self) -> list[str]:
        """Get list of all stored keys.

        Returns
        -------
        list[str]
            List of all keys currently stored in memory
        """
        return list(self.storage.keys())

    def has_key(self, key: str) -> bool:
        """Check if a key exists in storage.

        Parameters
        ----------
        key : str
            The key to check for existence

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        return key in self.storage

    def size(self) -> int:
        """Get the number of stored items.

        Returns
        -------
        int
            Number of items currently stored in memory
        """
        return len(self.storage)

    async def ahealth_check(self) -> "HealthStatus":
        """Check health of in-memory storage.

        Returns
        -------
        HealthStatus
            Health status indicating storage is operational

        Examples
        --------
        Example usage::

            memory = InMemoryMemory()
            status = await memory.ahealth_check()
        """
        from hexdag.core.ports.healthcheck import HealthStatus

        try:
            # Test basic operations
            test_key = "__health_check__"
            await self.aset(test_key, "test")
            value = await self.aget(test_key)

            # Clean up test key
            if test_key in self.storage:
                del self.storage[test_key]

            if value != "test":
                return HealthStatus(
                    status="unhealthy",
                    adapter_name="InMemoryMemory",
                    error=Exception("Storage read/write verification failed"),
                )

            details: dict[str, Any] = {"size": len(self.storage)}
            if self.max_size is not None:
                usage_percent = (len(self.storage) / self.max_size) * 100
                details["max_size"] = self.max_size
                details["usage_percent"] = round(usage_percent, 1)

                if usage_percent > 90:
                    return HealthStatus(
                        status="degraded",
                        adapter_name="InMemoryMemory",
                        details=details,
                    )

            return HealthStatus(
                status="healthy",
                adapter_name="InMemoryMemory",
                details=details,
                latency_ms=0.1,  # In-memory is always fast
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="InMemoryMemory",
                error=e,
            )
