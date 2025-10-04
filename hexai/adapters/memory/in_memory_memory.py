"""In-memory implementation of Memory for testing purposes."""

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.ports.healthcheck import HealthStatus

from pydantic import BaseModel, Field

from hexai.core.configurable import ConfigurableAdapter
from hexai.core.ports.memory import Memory
from hexai.core.registry import adapter

__all__ = ["InMemoryMemory"]


@adapter(name="in_memory_memory", implements_port="memory")
class InMemoryMemory(Memory, ConfigurableAdapter):
    """In-memory implementation of Memory for testing.

    Features:
    - Key-value storage in memory
    - Access history tracking
    - Delay simulation
    - Size limits
    - Reset functionality
    """

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for In-Memory adapter."""

        delay_seconds: float = Field(
            default=0.0, ge=0.0, description="Artificial delay to simulate storage access latency"
        )
        max_size: int | None = Field(
            default=None, gt=0, description="Maximum number of items to store (None for unlimited)"
        )

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the in-memory storage.

        Args
        ----
            **kwargs: Configuration options
        """
        # Initialize config (accessible via self.config.field_name)
        ConfigurableAdapter.__init__(self, **kwargs)

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
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)

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
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)

        # Check size limit
        if (
            self.config.max_size is not None
            and key not in self.storage
            and len(self.storage) >= self.config.max_size
        ):
            raise MemoryError(f"Memory limit of {self.config.max_size} items exceeded")

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
        from hexai.core.ports.healthcheck import HealthStatus

        # Check if storage is functional
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

            # Check if approaching size limit
            details: dict[str, Any] = {"size": len(self.storage)}
            if self.config.max_size is not None:
                usage_percent = (len(self.storage) / self.config.max_size) * 100
                details["max_size"] = self.config.max_size
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
