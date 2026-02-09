"""In-memory vector store for RAG operations."""

import hashlib
import math
from typing import Any

from hexdag_plugins.storage.ports import VectorStorePort


class InMemoryVectorStore(VectorStorePort):
    """In-memory vector store for RAG operations.

    Stores text chunks with embeddings and provides similarity search.
    Uses simple cosine similarity for retrieval.

    Parameters
    ----------
    embedding_dim : int
        Dimension of embedding vectors (default: 384 for sentence-transformers)
    max_results : int
        Maximum number of results to return from search (default: 5)

    Examples
    --------
    Store and search documents::

        vector_store = InMemoryVectorStore(embedding_dim=384, max_results=5)

        # Add documents
        await vector_store.aadd_documents([
            {"text": "Python is a programming language", "id": "doc1"},
            {"text": "Machine learning uses algorithms", "id": "doc2"},
        ])

        # Search
        results = await vector_store.asearch("programming", top_k=2)
    """

    _hexdag_icon = "Boxes"
    _hexdag_color = "#8b5cf6"  # Purple for in-memory

    def __init__(
        self,
        embedding_dim: int = 384,
        max_results: int = 5,
    ) -> None:
        """Initialize vector store.

        Parameters
        ----------
        embedding_dim : int
            Dimension of embedding vectors (default: 384)
        max_results : int
            Maximum number of results to return from search (default: 5)
        """
        self._embedding_dim = embedding_dim
        self._max_results = max_results
        self._documents: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []

    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Add documents to the vector store.

        Parameters
        ----------
        documents : list[dict[str, Any]]
            Documents to add (must have 'text' field)
        embeddings : list[list[float]] | None
            Pre-computed embeddings (if None, uses simple hash-based embedding)

        Returns
        -------
        dict[str, Any]
            Result with count of added documents
        """
        if embeddings is None:
            # Generate simple embeddings if none provided
            embeddings = [self._simple_embedding(doc["text"]) for doc in documents]

        if len(documents) != len(embeddings):
            msg = "Number of documents must match number of embeddings"
            raise ValueError(msg)

        self._documents.extend(documents)
        self._embeddings.extend(embeddings)

        return {"added": len(documents), "total": len(self._documents)}

    async def asearch(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Parameters
        ----------
        query : str
            Query text
        query_embedding : list[float] | None
            Pre-computed query embedding (if None, generates from query text)
        top_k : int | None
            Number of results to return (uses max_results if None)
        filter_metadata : dict[str, Any] | None
            Optional metadata filters (not implemented for in-memory store)

        Returns
        -------
        list[dict[str, Any]]
            Top-k most similar documents with similarity scores
        """
        if not self._documents:
            return []

        if query_embedding is None:
            query_embedding = self._simple_embedding(query)

        k = top_k if top_k is not None else self._max_results

        # Return empty list if top_k is explicitly 0
        if k == 0:
            return []

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []

        for idx, score in similarities[:k]:
            result = self._documents[idx].copy()
            result["similarity_score"] = score
            results.append(result)

        return results

    async def aclear(self) -> dict[str, Any]:
        """Clear all documents from the store.

        Returns
        -------
        dict[str, Any]
            Result with count of removed documents
        """
        count = len(self._documents)
        self._documents.clear()
        self._embeddings.clear()
        return {"removed": count}

    async def adelete(self, ids: list[str]) -> dict[str, Any]:
        """Delete documents by ID.

        Parameters
        ----------
        ids : list[str]
            List of document IDs to delete

        Returns
        -------
        dict[str, Any]
            Result with count of deleted documents
        """
        deleted_count = 0
        indices_to_remove = []

        # Find indices of documents to delete
        for i, doc in enumerate(self._documents):
            if doc.get("id") in ids:
                indices_to_remove.append(i)
                deleted_count += 1

        # Remove in reverse order to maintain indices
        for idx in reversed(indices_to_remove):
            del self._documents[idx]
            del self._embeddings[idx]

        return {"deleted": deleted_count}

    async def acount(self) -> int:
        """Get the number of documents in the vector store.

        Returns
        -------
        int
            Number of documents currently stored
        """
        return len(self._documents)

    async def aget_stats(self) -> dict[str, Any]:
        """Get vector store statistics.

        Returns
        -------
        dict[str, Any]
            Statistics about stored documents
        """
        return {
            "document_count": len(self._documents),
            "embedding_dim": self._embedding_dim,
            "max_results": self._max_results,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics (sync version for backwards compatibility).

        Returns
        -------
        dict[str, Any]
            Statistics about stored documents
        """
        return {
            "document_count": len(self._documents),
            "embedding_dim": self._embedding_dim,
            "max_results": self._max_results,
        }

    def _simple_embedding(self, text: str) -> list[float]:
        """Generate a simple hash-based embedding.

        This is a placeholder for production embedding models like
        sentence-transformers or OpenAI embeddings.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        list[float]
            Embedding vector
        """
        # Use multiple hash functions to create vector
        dim = self._embedding_dim
        vector = []

        # Normalize text
        normalized = text.lower().strip()

        for i in range(dim):
            # Create different seeds for hash
            seed = f"{normalized}_{i}"
            hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
            # Normalize to [-1, 1]
            vector.append((hash_val % 1000) / 500 - 1)

        return vector

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Parameters
        ----------
        vec1 : list[float]
            First vector
        vec2 : list[float]
            Second vector

        Returns
        -------
        float
            Cosine similarity score [-1, 1]
        """
        if len(vec1) != len(vec2):
            msg = "Vectors must have same dimension"
            raise ValueError(msg)

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def __repr__(self) -> str:
        """String representation."""
        return f"InMemoryVectorStore(documents={len(self._documents)})"
