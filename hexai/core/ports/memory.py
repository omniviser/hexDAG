"""Port interface for Long Term Memory."""

from typing import Any, Protocol


class LongTermMemory(Protocol):
    """Protocol for long-term memory storage and retrieval."""

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

    async def aset(self, key: str, value: Any) -> None:
        """Store a value in long-term memory asynchronously.

        Args
        ----
            key: The key to store under
            value: The value to store
        """
        pass
