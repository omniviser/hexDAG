"""Port interface definitions for Embedding generation.

.. deprecated:: 0.5.0
    The standalone ``embedding`` port is deprecated in favor of the unified ``llm`` port
    with ``SupportsEmbedding`` protocol. Use ``LLM`` port and implement ``SupportsEmbedding``
    instead. This port will be removed in version 1.0.0.

    Migration guide:
        Old (deprecated)::

            from hexdag.core.ports.embedding import Embedding

            @adapter("embedding", name="my_embedder")
            class MyEmbeddingAdapter(Embedding):
                async def aembed(self, text: str) -> list[float]:
                    ...

        New (recommended)::

            from hexdag.core.ports.llm import LLM, SupportsEmbedding

            @adapter("llm", name="my_embedder")
            class MyEmbeddingAdapter(LLM, SupportsEmbedding):
                async def aresponse(self, messages):
                    # For pure embedding adapters, can raise NotImplementedError
                    raise NotImplementedError("This is an embedding-only adapter")

                async def aembed(self, text: str) -> list[float]:
                    ...
"""

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus

type ImageInput = str | bytes

# Issue deprecation warning when module is imported
warnings.warn(
    "The 'embedding' port is deprecated and will be removed in version 0.5.0. "
    "Use 'llm' port with 'SupportsEmbedding' protocol instead. "
    "See hexdag.core.ports.llm.SupportsEmbedding for details.",
    DeprecationWarning,
    stacklevel=2,
)


@runtime_checkable
class Embedding(Protocol):
    """Port interface for Embedding generation.

    .. deprecated:: 0.5.0
        Use :class:`hexdag.core.ports.llm.LLM` with
        :class:`hexdag.core.ports.llm.SupportsEmbedding` protocol instead.

    Embeddings provide vector representations of text and images that capture semantic meaning.
    Implementations may use various backends (OpenAI, local models, etc.) but must
    provide the aembed method for generating embeddings from text.

    Required Methods
    ----------------
    - aembed(text: str): Generate embedding for text
    - aembed_image(image: ImageInput): Generate embedding for image

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify embedding API connectivity and availability
    - aembed_batch(): Batch embedding generation for efficiency
    - aembed_image_batch(): Batch image embedding generation for efficiency
    """

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input (async).

        Args
        ----
            text: Text string to embed

        Returns
        -------
            List of floats representing the embedding vector

        Examples
        --------
        Single text embedding::

            embedding = await adapter.aembed("Hello, world!")
            # Returns: [0.123, -0.456, 0.789, ...]
        """
        ...

    @abstractmethod
    async def aembed_image(self, image: ImageInput) -> list[float]:
        """Generate embedding vector for a single image input (async).

        Args
        ----
            image: Image to embed, either as:
                - str: File path to image or base64-encoded image data
                - bytes: Raw image bytes

        Returns
        -------
            List of floats representing the embedding vector

        Examples
        --------
        Image embedding from file path::

            embedding = await adapter.aembed_image("/path/to/image.jpg")
            # Returns: [0.123, -0.456, 0.789, ...]

        Image embedding from bytes::

            with open("image.jpg", "rb") as f:
                image_bytes = f.read()
            embedding = await adapter.aembed_image(image_bytes)
            # Returns: [0.123, -0.456, 0.789, ...]

        Image embedding from base64::

            base64_image = "data:image/jpeg;base64,/9j/4AAQ..."
            embedding = await adapter.aembed_image(base64_image)
            # Returns: [0.123, -0.456, 0.789, ...]
        """
        ...

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently (optional).

        This method enables batch processing for improved performance when
        embedding multiple texts. If not implemented, the framework will
        fall back to sequential calls to aembed().

        Args
        ----
            texts: List of text strings to embed

        Returns
        -------
            List of embedding vectors, one per input text

        Examples
        --------
        Batch embedding::

            texts = ["Hello", "World", "AI"]
            embeddings = await adapter.aembed_batch(texts)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        ...

    async def aembed_image_batch(self, images: list[ImageInput]) -> list[list[float]]:
        """Generate embeddings for multiple images efficiently (optional).

        This method enables batch processing for improved performance when
        embedding multiple images. If not implemented, the framework will
        fall back to sequential calls to aembed_image().

        Args
        ----
            images: List of images to embed, each can be:
                - str: File path to image or base64-encoded image data
                - bytes: Raw image bytes

        Returns
        -------
            List of embedding vectors, one per input image

        Examples
        --------
        Batch image embedding::

            images = [
                "/path/to/image1.jpg",
                "/path/to/image2.png",
                image_bytes
            ]
            embeddings = await adapter.aembed_image_batch(images)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        ...

    async def ahealth_check(self) -> "HealthStatus":
        """Check embedding adapter health and connectivity (optional).

        Adapters should verify:
        - API connectivity to the embedding service
        - Model availability
        - Authentication status
        - Rate limit status (if applicable)

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about connectivity and availability

        Examples
        --------
        OpenAI adapter health check::

            status = await openai_adapter.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.latency_ms  # Time taken for health check
            status.details  # {"model": "text-embedding-3-small"}
        """
        ...
