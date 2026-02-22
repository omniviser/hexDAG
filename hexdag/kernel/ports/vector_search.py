"""Port interface for vector similarity search capabilities."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsVectorSearch(Protocol):
    """Protocol for adapters that support vector similarity search.

    This protocol enables semantic search capabilities using vector embeddings,
    commonly used with vector databases like Pinecone, Weaviate, pgvector, etc.
    """

    @abstractmethod
    async def avector_search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        Args
        ----
            collection: Name of the vector collection/table to search
            query_vector: Query embedding vector (must match collection dimension)
            top_k: Number of nearest neighbors to return (default: 10)
            filters: Optional metadata filters to apply
            include_metadata: Whether to include metadata in results (default: True)
            include_vectors: Whether to include vectors in results (default: False)

        Returns
        -------
            List of search results with structure::

                [
                    {
                        "id": str,              # Document/vector ID
                        "score": float,         # Similarity score (0-1 or distance)
                        "metadata": dict,       # Document metadata (if include_metadata=True)
                        "vector": list[float]   # Vector embedding (if include_vectors=True)
                    }
                ]

        Raises
        ------
        ValueError
            If collection doesn't exist or query_vector dimension mismatch

        Examples
        --------
        Basic semantic search::

            results = await db.avector_search(
                collection="documents",
                query_vector=[0.1, 0.2, ...],  # 768-dim embedding
                top_k=5
            )

        Search with metadata filtering::

            results = await db.avector_search(
                collection="products",
                query_vector=embedding,
                top_k=10,
                filters={"category": "electronics", "price_range": "100-500"}
            )
        """
        ...

    @abstractmethod
    async def avector_upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert or update vectors in a collection.

        Args
        ----
            collection: Name of the vector collection/table
            vectors: List of vectors to upsert with structure::

                [
                    {
                        "id": str,                  # Unique document ID
                        "vector": list[float],      # Embedding vector
                        "metadata": dict            # Optional metadata
                    }
                ]

        Returns
        -------
            Dictionary with upsert statistics::

                {
                    "upserted_count": int,
                    "updated_count": int,
                    "failed_count": int
                }

        Raises
        ------
        ValueError
            If collection doesn't exist or vector dimensions mismatch

        Examples
        --------
        Insert document embeddings::

            await db.avector_upsert(
                collection="documents",
                vectors=[
                    {
                        "id": "doc1",
                        "vector": [0.1, 0.2, ...],
                        "metadata": {"title": "Document 1", "author": "John"}
                    }
                ]
            )
        """
        ...

    @abstractmethod
    async def avector_delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delete vectors from a collection.

        Args
        ----
            collection: Name of the vector collection/table
            ids: Optional list of document IDs to delete
            filters: Optional metadata filters for bulk deletion

        Returns
        -------
            Dictionary with deletion statistics::

                {
                    "deleted_count": int
                }

        Raises
        ------
        ValueError
            If neither ids nor filters are provided

        Examples
        --------
        Delete by IDs::

            await db.avector_delete(
                collection="documents",
                ids=["doc1", "doc2"]
            )

        Delete by metadata filter::

            await db.avector_delete(
                collection="documents",
                filters={"category": "archived"}
            )
        """
        ...
