"""In-memory implementation of Memory for testing purposes."""

import asyncio
from typing import Any

from hexai.adapters.configs import InMemoryMemoryConfig
from hexai.core.ports.memory import Memory
from hexai.core.registry import adapter

__all__ = ["InMemoryMemory"]


@adapter(name="in_memory_memory", implements_port="memory", namespace="plugin")
class InMemoryMemory(Memory):
    """In-memory implementation of Memory for testing.

    Features:
    - Key-value storage in memory
    - Access history tracking
    - Delay simulation
    - Size limits
    - Reset functionality
    """

    def __init__(self, config: InMemoryMemoryConfig | None = None) -> None:
        """Initialize the in-memory storage.

        Args
        ----
            config: Configuration for the memory adapter
        """
        if config is None:
            config = InMemoryMemoryConfig()

        self.storage: dict[str, Any] = {}
        self.delay_seconds = config.delay_seconds
        self.max_size = config.max_size
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
        self.access_history.append(
            {
                "operation": "get",
                "key": key,
                "found": key in self.storage,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        return result

    async def aset(self, key: str, value: Any) -> None:
        """Store a value in memory.

        Args
        ----
            key: The key to store under
            value: The value to store

        Raises
        ------
            MemoryError: If max_size is exceeded
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
        self.access_history.append(
            {
                "operation": "set",
                "key": key,
                "value_type": type(value).__name__,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def clear(self) -> None:
        """Clear all stored data."""
        self.storage.clear()

    def reset(self) -> None:
        """Reset the memory to initial state, clearing both data and history."""
        self.storage.clear()
        self.access_history.clear()

    def get_access_history(self) -> list[dict[str, Any]]:
        """Get the history of all memory access operations."""
        return self.access_history.copy()

    def get_stored_keys(self) -> list[str]:
        """Get list of all stored keys."""
        return list(self.storage.keys())

    def has_key(self, key: str) -> bool:
        """Check if a key exists in storage."""
        return key in self.storage

    def size(self) -> int:
        """Get the number of stored items."""
        return len(self.storage)
