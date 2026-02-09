"""Vector store port interface for RAG operations."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


@runtime_checkable
class VectorStorePort(Protocol):
    """Port interface for vector store adapters.

    Vector stores provide semantic search capabilities by storing text embeddings
    and enabling similarity-based retrieval. This port abstracts different vector
    database backends (in-memory, PostgreSQL pgvector, ChromaDB, etc.).

    All vector store implementations must provide:
    - Document storage with embeddings
    - Similarity search
    - Document management (count, clear)

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - adelete(ids): Delete documents by ID
    - aget_stats(): Get storage statistics
    - asetup(): Initialize database/connections
    - aclose(): Clean up resources
    - ahealth_check(): Verify adapter health and connectivity
    """

    @abstractmethod
    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None | dict[str, Any]:
        """Add documents to the vector store.

        Args
        ----
            documents: List of documents with required 'text' field and optional 'id', 'metadata'
            embeddings: Optional pre-computed embeddings (if None, adapter may compute them)

        Returns
        -------
            Optional dict with operation results (e.g., {'added': 5, 'ids': [...]})

        Examples
        --------
        Basic usage::

            await vector_store.aadd_documents([
                {"text": "Python programming", "id": "doc1"},
                {"text": "Machine learning", "id": "doc2"},
            ])

        With embeddings::

            embeddings = await embedder.aembed_texts(["text1", "text2"])
            await vector_store.aadd_documents(documents, embeddings=embeddings)
        """
        ...

    @abstractmethod
    async def asearch(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args
        ----
            query: Search query text
            query_embedding: Optional pre-computed query embedding
            top_k: Number of results to return (adapter may have default)
            filter_metadata: Optional metadata filters (e.g., {"category": "docs"})

        Returns
        -------
            List of matching documents with 'id', 'text', 'score', and optional 'metadata'
            Scores should be similarity scores (higher = more similar)

        Examples
        --------
        Basic search::

            results = await vector_store.asearch("Python tutorial", top_k=5)
            for doc in results:
                print(f"{doc['text']} (score: {doc['score']:.3f})")

        With metadata filter::

            results = await vector_store.asearch(
                "tutorial",
                filter_metadata={"category": "programming"}
            )
        """
        ...

    @abstractmethod
    async def aclear(self) -> None | dict[str, Any]:
        """Clear all documents from the vector store.

        Returns
        -------
            Optional dict with operation results (e.g., {'removed': 100})

        Examples
        --------
        ::

            await vector_store.aclear()
            count = await vector_store.acount()
            assert count == 0
        """
        ...

    @abstractmethod
    async def acount(self) -> int:
        """Get the number of documents in the vector store.

        Returns
        -------
            Number of documents currently stored

        Examples
        --------
        ::

            count = await vector_store.acount()
            print(f"Store contains {count} documents")
        """
        ...

    # Optional methods for enhanced functionality

    async def adelete(self, ids: list[str]) -> None | dict[str, Any]:
        """Delete documents by ID (optional).

        Args
        ----
            ids: List of document IDs to delete

        Returns
        -------
            Optional dict with operation results (e.g., {'deleted': 3})

        Examples
        --------
        ::

            await vector_store.adelete(["doc1", "doc2"])
        """
        ...

    async def aget_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store (optional).

        Returns
        -------
            Dictionary with statistics like document count, storage size, etc.

        Examples
        --------
        ::

            stats = await vector_store.aget_stats()
            print(f"Documents: {stats['document_count']}")
            print(f"Size: {stats.get('storage_size_mb', 'N/A')} MB")
        """
        ...

    async def asetup(self) -> None:
        """Initialize the vector store (optional).

        This method can be used for one-time setup like:
        - Creating database tables/indexes
        - Establishing connections
        - Loading models

        Examples
        --------
        ::

            pgvector = PgVectorAdapter(connection_string="...")
            await pgvector.asetup()  # Creates tables and indexes
        """
        ...

    async def aclose(self) -> None:
        """Close connections and clean up resources (optional).

        Examples
        --------
        ::

            await vector_store.aclose()
        """
        ...

    async def ahealth_check(self) -> "HealthStatus":
        """Check vector store health and connectivity (optional).

        Adapters should verify:
        - Storage backend connectivity (database, file system, etc.)
        - Read/write operations
        - Index health and performance
        - Storage capacity/availability

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about storage backend

        Examples
        --------
        ::

            status = await vector_store.ahealth_check()
            if status.status == "healthy":
                print(f"Store is healthy (latency: {status.latency_ms}ms)")
            else:
                print(f"Store is {status.status}: {status.error}")
        """
        ...
