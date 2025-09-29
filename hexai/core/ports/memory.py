"""Port interface for Long Term Memory."""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from hexai.core.registry.decorators import port


@port(
    name="memory",
    namespace="core",
)
@runtime_checkable
class Memory(Protocol):
    """Protocol for long-term memory storage and retrieval."""

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
