"""Mock Embedding implementation for testing purposes."""

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

from hexdag.core.registry import adapter

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


@adapter(name="mock_embedding", implements_port="embedding", namespace="plugin")
class MockEmbedding:
    """Mock implementation of the Embedding interface for testing.

    This mock generates deterministic embeddings based on the input text's hash,
    making tests predictable and reproducible. It also provides utilities for
    testing like call inspection and configurable delays.
    """

    # Type annotations for attributes
    delay_seconds: float
    dimensions: int
    call_count: int
    last_texts: list[str]
    should_raise: bool

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration.

        Args
        ----
            **kwargs: Configuration options
                - dimensions: Embedding vector dimensions (default: 1536)
                - delay_seconds: Delay before returning embeddings (default: 0.0)
        """
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)
        self.dimensions = kwargs.get("dimensions", 1536)

        # Non-config state
        self.call_count = 0
        self.last_texts: list[str] = []
        self.should_raise = False

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding vector from text.

        Uses text hash to generate consistent embeddings for the same input.
        """
        # Use hash to generate deterministic values (not for security)
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # noqa: S324

        # Generate embedding values from hash
        embedding = []
        for i in range(self.dimensions):
            # Use chunks of the hash to generate values
            chunk_idx = (i * 4) % len(text_hash)
            chunk = text_hash[chunk_idx : chunk_idx + 4]
            # Convert to float in range [-1, 1]
            value = (int(chunk, 16) / 65535.0) * 2 - 1
            embedding.append(value)

        return embedding

    async def aembed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input.

        Parameters
        ----------
        text : str
            Text string to embed

        Returns
        -------
        list[float]
            Mock embedding vector

        Raises
        ------
        Exception
            When should_raise is True for testing error conditions
        """
        self.last_texts = [text]

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_raise:
            raise Exception("Mock Embedding error for testing")

        self.call_count += 1
        return self._generate_embedding(text)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Parameters
        ----------
        texts : list[str]
            List of text strings to embed

        Returns
        -------
        list[list[float]]
            List of mock embedding vectors

        Raises
        ------
        Exception
            When should_raise is True for testing error conditions
        """
        self.last_texts = texts

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_raise:
            raise Exception("Mock Embedding batch error for testing")

        self.call_count += len(texts)
        return [self._generate_embedding(text) for text in texts]

    async def ahealth_check(self) -> "HealthStatus":
        """Health check for Mock Embedding (always healthy)."""
        from hexdag.core.ports.healthcheck import HealthStatus

        return HealthStatus(
            status="healthy",
            adapter_name="MockEmbedding",
            latency_ms=0.1,
            details={"dimensions": self.dimensions},
        )

    # Testing utilities (not part of the Embedding port interface)
    def reset(self) -> None:
        """Reset the mock state for testing."""
        self.call_count = 0
        self.last_texts = []
        self.should_raise = False
