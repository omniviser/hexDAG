"""In-memory implementation of LongTermMemory for testing purposes."""

import asyncio
from typing import Any

from hexai.core.ports.memory import LongTermMemory


class InMemoryMemory(LongTermMemory):
    """In-memory implementation of LongTermMemory for testing.

    Features:
    - Key-value storage in memory
    - Access history tracking
    - Delay simulation
    - Expiration support
    - Reset functionality
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        """Initialize the in-memory storage.

        Args
        ----
            delay_seconds: Artificial delay to simulate network/disk access.
        """
        self.storage: dict[str, Any] = {}
        self.delay_seconds = delay_seconds
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
        """
        # Simulate access delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

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
