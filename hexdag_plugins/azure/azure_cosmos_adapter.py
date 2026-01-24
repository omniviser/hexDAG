"""Azure Cosmos DB adapter for hexDAG framework.

Provides persistent memory and state storage for agents and pipelines.
"""

import time
from typing import Any

from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.ports.memory import Memory
from hexdag.core.registry import adapter


@adapter(
    "memory",
    name="azure_cosmos",
    secrets={
        "key": "AZURE_COSMOS_KEY",
    },
)
class AzureCosmosAdapter(Memory):
    """Azure Cosmos DB adapter for agent memory and pipeline state.

    Provides persistent, scalable storage for agent memory, conversation
    history, and pipeline execution state using Azure Cosmos DB.

    Parameters
    ----------
    endpoint : str
        Azure Cosmos DB endpoint URL
    key : str
        Azure Cosmos DB primary or secondary key (auto-resolved from AZURE_COSMOS_KEY)
    database_name : str
        Database name (default: "hexdag")
    container_name : str
        Container name (default: "memory")
    partition_key : str
        Partition key path (default: "/agent_id")
    use_managed_identity : bool
        Use Managed Identity instead of key auth (default: False)
    throughput : int
        Container throughput in RU/s (default: 400)

    Examples
    --------
    YAML configuration::

        spec:
          ports:
            memory:
              adapter: azure_cosmos
              config:
                endpoint: ${AZURE_COSMOS_ENDPOINT}
                database_name: "hexdag"
                container_name: "agent_memory"

    Python usage::

        from hexdag_plugins.azure import AzureCosmosAdapter

        adapter = AzureCosmosAdapter(
            endpoint="https://my-cosmos.documents.azure.com:443/",
            key="...",  # or auto-resolved from AZURE_COSMOS_KEY
            database_name="hexdag",
            container_name="agent_memory"
        )

        # Store agent memory
        await adapter.astore("agent-123", {"context": "...", "history": [...]})

        # Retrieve memory
        memory = await adapter.aretrieve("agent-123")

        # Search memories
        results = await adapter.asearch("user query", top_k=5)
    """

    def __init__(
        self,
        endpoint: str,
        key: str | None = None,
        database_name: str = "hexdag",
        container_name: str = "memory",
        partition_key: str = "/agent_id",
        use_managed_identity: bool = False,
        throughput: int = 400,
    ):
        """Initialize Azure Cosmos DB adapter.

        Args
        ----
            endpoint: Azure Cosmos DB endpoint URL
            key: Cosmos DB key (or use managed identity)
            database_name: Database name (default: "hexdag")
            container_name: Container name (default: "memory")
            partition_key: Partition key path (default: "/agent_id")
            use_managed_identity: Use Managed Identity (default: False)
            throughput: Container throughput RU/s (default: 400)
        """
        self.endpoint = endpoint
        self.key = key
        self.database_name = database_name
        self.container_name = container_name
        self.partition_key = partition_key
        self.use_managed_identity = use_managed_identity
        self.throughput = throughput

        self._client = None
        self._container = None

    async def _get_container(self):
        """Get or create Cosmos DB container."""
        if self._container is None:
            try:
                from azure.cosmos import PartitionKey
                from azure.cosmos.aio import CosmosClient
            except ImportError as e:
                raise ImportError(
                    "Azure Cosmos SDK not installed. Install with: pip install azure-cosmos"
                ) from e

            if self.use_managed_identity:
                try:
                    from azure.identity.aio import DefaultAzureCredential

                    credential = DefaultAzureCredential()
                    self._client = CosmosClient(self.endpoint, credential=credential)
                except ImportError as e:
                    raise ImportError(
                        "Azure Identity SDK not installed. Install with: pip install azure-identity"
                    ) from e
            else:
                if not self.key:
                    raise ValueError("key is required when use_managed_identity=False")
                self._client = CosmosClient(self.endpoint, credential=self.key)

            # Create database if not exists
            database = await self._client.create_database_if_not_exists(id=self.database_name)

            # Create container if not exists
            self._container = await database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path=self.partition_key),
                offer_throughput=self.throughput,
            )

        return self._container

    async def astore(
        self,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store data in Cosmos DB.

        Args
        ----
            key: Unique identifier for the data
            value: Data to store
            metadata: Optional metadata
        """
        container = await self._get_container()

        # Extract agent_id from key for partitioning
        agent_id = key.split(":")[0] if ":" in key else key

        document = {
            "id": key,
            "agent_id": agent_id,
            "data": value,
            "metadata": metadata or {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        await container.upsert_item(document)

    async def aretrieve(self, key: str) -> dict[str, Any] | None:
        """Retrieve data from Cosmos DB.

        Args
        ----
            key: Unique identifier for the data

        Returns
        -------
            Stored data or None if not found
        """
        container = await self._get_container()
        agent_id = key.split(":")[0] if ":" in key else key

        try:
            item = await container.read_item(item=key, partition_key=agent_id)
            return item.get("data")
        except Exception:
            return None

    async def adelete(self, key: str) -> bool:
        """Delete data from Cosmos DB.

        Args
        ----
            key: Unique identifier for the data

        Returns
        -------
            True if deleted, False if not found
        """
        container = await self._get_container()
        agent_id = key.split(":")[0] if ":" in key else key

        try:
            await container.delete_item(item=key, partition_key=agent_id)
            return True
        except Exception:
            return False

    async def alist(self, prefix: str | None = None) -> list[str]:
        """List all keys in memory.

        Args
        ----
            prefix: Optional prefix to filter keys

        Returns
        -------
            List of keys
        """
        container = await self._get_container()

        if prefix:
            query = f"SELECT c.id FROM c WHERE STARTSWITH(c.id, '{prefix}')"
        else:
            query = "SELECT c.id FROM c"

        items = container.query_items(query=query, enable_cross_partition_query=True)
        return [item["id"] async for item in items]

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories by content.

        Note: This performs a simple text search. For vector similarity search,
        consider using Azure Cognitive Search integration.

        Args
        ----
            query: Search query string
            top_k: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns
        -------
            List of matching documents
        """
        container = await self._get_container()

        # Build SQL query with CONTAINS for text search
        sql_query = f"""
            SELECT TOP {top_k} c.id, c.data, c.metadata, c.created_at
            FROM c
            WHERE CONTAINS(LOWER(c.data), LOWER('{query}'))
        """

        if filter_metadata:
            for key, value in filter_metadata.items():
                sql_query += f" AND c.metadata.{key} = '{value}'"

        sql_query += " ORDER BY c.created_at DESC"

        items = container.query_items(query=sql_query, enable_cross_partition_query=True)
        return [item async for item in items]

    async def astore_conversation(
        self,
        agent_id: str,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> None:
        """Store conversation history.

        Args
        ----
            agent_id: Agent identifier
            messages: List of message dicts with 'role' and 'content'
            session_id: Optional session identifier
        """
        key = f"{agent_id}:conversation:{session_id or 'default'}"
        await self.astore(key, {"messages": messages}, {"type": "conversation"})

    async def aretrieve_conversation(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Retrieve conversation history.

        Args
        ----
            agent_id: Agent identifier
            session_id: Optional session identifier

        Returns
        -------
            List of messages
        """
        key = f"{agent_id}:conversation:{session_id or 'default'}"
        data = await self.aretrieve(key)
        return data.get("messages", []) if data else []

    async def aclear_agent(self, agent_id: str) -> int:
        """Clear all data for an agent.

        Args
        ----
            agent_id: Agent identifier

        Returns
        -------
            Number of items deleted
        """
        keys = await self.alist(prefix=agent_id)
        count = 0
        for key in keys:
            if await self.adelete(key):
                count += 1
        return count

    async def ahealth_check(self) -> HealthStatus:
        """Check Azure Cosmos DB connectivity.

        Returns
        -------
            HealthStatus with connectivity details
        """
        try:
            start_time = time.time()
            container = await self._get_container()

            # Simple query to verify connectivity
            query = "SELECT VALUE COUNT(1) FROM c"
            items = container.query_items(query=query, enable_cross_partition_query=True)
            count = 0
            async for item in items:
                count = item

            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                status="healthy",
                adapter_name="AzureCosmos",
                latency_ms=latency_ms,
                details={
                    "endpoint": self.endpoint,
                    "database": self.database_name,
                    "container": self.container_name,
                    "document_count": count,
                },
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="AzureCosmos",
                latency_ms=0.0,
                details={"error": str(e)},
            )

    async def aclose(self) -> None:
        """Close the Cosmos DB client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._container = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter configuration (excluding secrets)."""
        return {
            "endpoint": self.endpoint,
            "database_name": self.database_name,
            "container_name": self.container_name,
            "partition_key": self.partition_key,
            "use_managed_identity": self.use_managed_identity,
            "throughput": self.throughput,
        }
