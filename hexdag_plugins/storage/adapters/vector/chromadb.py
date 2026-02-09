"""ChromaDB vector store adapter for RAG plugin.

ChromaDB is an open-source embedding database that provides:
- Easy local development and deployment
- Built-in embedding models
- Persistent storage
- Cloud deployment option

Installation:
    pip install chromadb
"""

from typing import Any, Literal

from hexdag_plugins.storage.ports import VectorStorePort

# Convention: ChromaDB distance metric options for dropdown menus in Studio UI
ChromaDistanceMetric = Literal["cosine", "l2", "ip"]

# Convention: ChromaDB embedding function options for dropdown menus in Studio UI
ChromaEmbeddingFunction = Literal["default", "sentence-transformers", "openai"]


class ChromaDBAdapter(VectorStorePort):
    """ChromaDB vector store adapter.

    Provides persistent vector storage with built-in embedding support.

    Parameters
    ----------
    collection_name : str
        Name of the ChromaDB collection (default: "hexdag_documents")
    persist_directory : str | None
        Directory for persistent storage (None for in-memory)
    embedding_function : str
        Embedding function to use (default: "default")
        Options: "default", "sentence-transformers", "openai"
    distance_metric : str
        Distance metric for similarity (default: "cosine")
        Options: "cosine", "l2", "ip" (inner product)

    Examples
    --------
    >>> # In-memory ChromaDB
    >>> store = ChromaDBAdapter(collection_name="docs")
    >>> await store.aadd_documents([{"text": "Python programming"}])
    >>> results = await store.asearch("Python", top_k=5)

    >>> # Persistent ChromaDB
    >>> store = ChromaDBAdapter(
    ...     collection_name="docs",
    ...     persist_directory="./chroma_db"
    ... )
    """

    _hexdag_icon = "Boxes"
    _hexdag_color = "#f59e0b"  # Amber for vector stores

    def __init__(
        self,
        collection_name: str = "hexdag_documents",
        persist_directory: str | None = None,
        embedding_function: ChromaEmbeddingFunction = "default",
        distance_metric: ChromaDistanceMetric = "cosine",
    ) -> None:
        """Initialize ChromaDB adapter.

        Parameters
        ----------
        collection_name : str
            Name of the ChromaDB collection (default: "hexdag_documents")
        persist_directory : str | None
            Directory for persistent storage (None for in-memory)
        embedding_function : str
            Embedding function to use (default: "default")
        distance_metric : str
            Distance metric: "cosine", "l2", or "ip" (default: "cosine")
        """
        if distance_metric not in ("cosine", "l2", "ip"):
            msg = f"Invalid distance_metric: {distance_metric}. Must be one of: cosine, l2, ip"
            raise ValueError(msg)

        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._distance_metric = distance_metric
        self._client = None
        self._collection = None

    async def asetup(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            msg = "ChromaDB not installed. Install with: pip install chromadb"
            raise ImportError(msg) from e

        # Create client
        if self._persist_directory:
            settings = Settings(
                persist_directory=self._persist_directory,
                anonymized_telemetry=False,
            )
            self._client = chromadb.Client(settings)
        else:
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"distance_metric": self._distance_metric},
        )

    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add documents to ChromaDB.

        ChromaDB can generate embeddings automatically if not provided.

        Parameters
        ----------
        documents : list[dict[str, Any]]
            List of documents with 'text' and optional 'metadata'
        embeddings : list[list[float]] | None
            Optional pre-computed embeddings (if None, ChromaDB generates)
        """
        if not self._collection:
            await self.asetup()

        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

        if embeddings:
            # Use provided embeddings
            self._collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            # Let ChromaDB generate embeddings
            self._collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

    async def asearch(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents in ChromaDB.

        Parameters
        ----------
        query : str
            Search query text
        query_embedding : list[float] | None
            Optional pre-computed query embedding
        top_k : int | None
            Number of results to return (default: 5)
        filter_metadata : dict[str, Any] | None
            Optional metadata filters (ChromaDB where clause)

        Returns
        -------
        list[dict[str, Any]]
            List of matching documents with scores and metadata
        """
        if not self._collection:
            await self.asetup()

        k = top_k or 5

        # Build where clause from filter_metadata
        where = None
        if filter_metadata:
            where = filter_metadata

        if query_embedding:
            # Use provided embedding
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
            )
        else:
            # Let ChromaDB embed the query
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
            )

        # Format results
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            }
            for i in range(len(results["ids"][0]))
        ]

    async def aclear(self) -> None:
        """Clear all documents from the collection."""
        if not self._collection:
            await self.asetup()

        # Delete and recreate collection
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"distance_metric": self._distance_metric},
        )

    async def acount(self) -> int:
        """Get the number of documents in the collection."""
        if not self._collection:
            await self.asetup()

        return self._collection.count()

    async def adelete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Parameters
        ----------
        ids : list[str]
            List of document IDs to delete
        """
        if not self._collection:
            await self.asetup()

        self._collection.delete(ids=ids)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChromaDBAdapter(collection={self._collection_name}, "
            f"persist={self._persist_directory is not None})"
        )
