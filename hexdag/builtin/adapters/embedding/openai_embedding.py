"""OpenAI adapter for embedding generation."""

from typing import Any

from openai import AsyncOpenAI

from hexdag.core.logging import get_logger
from hexdag.core.registry import adapter
from hexdag.core.types import RetryCount, TimeoutSeconds

logger = get_logger(__name__)


@adapter(
    name="openai",
    implements_port="embedding",
    namespace="core",
    description="OpenAI adapter for embedding generation",
    secrets={"api_key": "OPENAI_API_KEY"},
)
class OpenAIEmbeddingAdapter:
    """OpenAI implementation of the Embedding port.

    This adapter provides integration with OpenAI's embedding models through
    their API. It supports async operations and batch processing for efficient
    embedding generation.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit parameter: OpenAIEmbeddingAdapter(api_key="sk-...")
    2. Environment variable: OPENAI_API_KEY
    3. Memory port (orchestrator): secret:OPENAI_API_KEY
    """

    def __init__(
        self,
        api_key: str,  # ← Auto-resolved by @adapter decorator
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        timeout: TimeoutSeconds = 60.0,
        max_retries: RetryCount = 2,
        **kwargs: Any,  # ← For extra params like organization, base_url
    ):
        """Initialize OpenAI embedding adapter.

        Parameters
        ----------
        api_key : str
            OpenAI API key (auto-resolved from OPENAI_API_KEY env var)
        model : str, default="text-embedding-3-small"
            OpenAI embedding model to use. Options:
            - "text-embedding-3-small" (1536 dimensions, cost-effective)
            - "text-embedding-3-large" (3072 dimensions, higher performance)
            - "text-embedding-ada-002" (1536 dimensions, legacy)
        dimensions : int | None, default=None
            Optional dimensionality reduction (only for text-embedding-3-* models)
        timeout : float, default=60.0
            Request timeout in seconds
        max_retries : int, default=2
            Maximum retry attempts
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self._extra_kwargs = kwargs  # Store extra params

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        if org := kwargs.get("organization"):
            client_kwargs["organization"] = org
        if base_url := kwargs.get("base_url"):
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

    async def aembed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input.

        Args
        ----
            text: Text string to embed

        Returns
        -------
            List of floats representing the embedding vector
        """
        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "input": text,
            }

            if self.dimensions is not None:
                request_params["dimensions"] = self.dimensions

            response = await self.client.embeddings.create(**request_params)

            if response.data and len(response.data) > 0:
                embedding: list[float] = response.data[0].embedding
                return embedding

            logger.warning("No embedding data in OpenAI response")
            return []

        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}", exc_info=True)
            raise

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args
        ----
            texts: List of text strings to embed

        Returns
        -------
            List of embedding vectors, one per input text
        """
        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "input": texts,
            }

            if self.dimensions is not None:
                request_params["dimensions"] = self.dimensions

            response = await self.client.embeddings.create(**request_params)

            if response.data:
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            logger.warning("No embedding data in OpenAI batch response")
            return [[] for _ in texts]

        except Exception as e:
            logger.error(f"OpenAI batch embedding API error: {e}", exc_info=True)
            raise

    async def ahealth_check(self) -> Any:
        """Check OpenAI embedding adapter health and connectivity.

        Returns
        -------
        HealthStatus
            Current health status with connectivity details
        """
        from hexdag.core.ports.healthcheck import HealthStatus

        try:
            # Try a minimal embedding request
            import time

            start = time.time()
            await self.aembed("test")
            latency_ms = (time.time() - start) * 1000

            return HealthStatus(
                status="healthy",
                adapter_name="OpenAIEmbeddingAdapter",
                latency_ms=latency_ms,
                details={"model": self.model, "dimensions": self.dimensions},
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                status="unhealthy",
                adapter_name="OpenAIEmbeddingAdapter",
                latency_ms=0.0,
                details={"error": str(e)},
            )
