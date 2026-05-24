"""Tests for Azure Cosmos DB adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag_plugins.azure.adapters.cosmos import AzureCosmosAdapter


@pytest.fixture
def cosmos_adapter():
    """Create Azure Cosmos adapter for testing."""
    return AzureCosmosAdapter(
        endpoint="https://test-cosmos.documents.azure.com:443/",
        key="test-key-123",
        database_name="test-db",
        container_name="test-container",
    )


@pytest.fixture
def cosmos_adapter_managed_identity():
    """Create Azure Cosmos adapter with managed identity."""
    return AzureCosmosAdapter(
        endpoint="https://test-cosmos.documents.azure.com:443/",
        database_name="test-db",
        container_name="test-container",
        use_managed_identity=True,
    )


@pytest.mark.asyncio
async def test_adapter_initialization(cosmos_adapter):
    """Test adapter initializes with correct parameters."""
    assert cosmos_adapter.endpoint == "https://test-cosmos.documents.azure.com:443/"
    assert cosmos_adapter.database_name == "test-db"
    assert cosmos_adapter.container_name == "test-container"
    assert cosmos_adapter.partition_key == "/agent_id"
    assert cosmos_adapter.use_managed_identity is False
    assert cosmos_adapter._container is None


@pytest.mark.asyncio
async def test_store_success(cosmos_adapter):
    """Test successful data storage."""
    mock_container = AsyncMock()
    mock_container.upsert_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        await cosmos_adapter.astore("agent-1", {"context": "test data"})

        mock_container.upsert_item.assert_called_once()
        call_args = mock_container.upsert_item.call_args
        document = call_args[0][0]
        assert document["id"] == "agent-1"
        assert document["agent_id"] == "agent-1"
        assert document["data"] == {"context": "test data"}


@pytest.mark.asyncio
async def test_store_with_composite_key(cosmos_adapter):
    """Test storage with composite key extracts agent_id."""
    mock_container = AsyncMock()
    mock_container.upsert_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        await cosmos_adapter.astore("agent-1:conversation:session-1", {"messages": []})

        call_args = mock_container.upsert_item.call_args
        document = call_args[0][0]
        assert document["id"] == "agent-1:conversation:session-1"
        assert document["agent_id"] == "agent-1"


@pytest.mark.asyncio
async def test_store_with_metadata(cosmos_adapter):
    """Test storage with metadata."""
    mock_container = AsyncMock()
    mock_container.upsert_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        await cosmos_adapter.astore("agent-1", {"data": "value"}, metadata={"type": "context"})

        call_args = mock_container.upsert_item.call_args
        document = call_args[0][0]
        assert document["metadata"] == {"type": "context"}


@pytest.mark.asyncio
async def test_retrieve_success(cosmos_adapter):
    """Test successful data retrieval."""
    mock_item = {"id": "agent-1", "data": {"context": "test"}}

    mock_container = AsyncMock()
    mock_container.read_item = AsyncMock(return_value=mock_item)

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        result = await cosmos_adapter.aretrieve("agent-1")

        assert result == {"context": "test"}
        mock_container.read_item.assert_called_once_with(item="agent-1", partition_key="agent-1")


@pytest.mark.asyncio
async def test_retrieve_not_found(cosmos_adapter):
    """Test retrieval returns None for missing key."""
    mock_container = AsyncMock()
    mock_container.read_item = AsyncMock(side_effect=Exception("NotFound"))

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        result = await cosmos_adapter.aretrieve("missing-key")

        assert result is None


@pytest.mark.asyncio
async def test_delete_success(cosmos_adapter):
    """Test successful deletion."""
    mock_container = AsyncMock()
    mock_container.delete_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        result = await cosmos_adapter.adelete("agent-1")

        assert result is True
        mock_container.delete_item.assert_called_once_with(item="agent-1", partition_key="agent-1")


@pytest.mark.asyncio
async def test_delete_not_found(cosmos_adapter):
    """Test deletion returns False for missing key."""
    mock_container = AsyncMock()
    mock_container.delete_item = AsyncMock(side_effect=Exception("NotFound"))

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        result = await cosmos_adapter.adelete("missing-key")

        assert result is False


@pytest.mark.asyncio
async def test_list_all_keys(cosmos_adapter):
    """Test listing all keys."""
    mock_items = [{"id": "key-1"}, {"id": "key-2"}]

    async def mock_iter():
        for item in mock_items:
            yield item

    mock_container = MagicMock()
    mock_container.query_items.return_value = mock_iter()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        keys = await cosmos_adapter.alist()

        assert keys == ["key-1", "key-2"]
        mock_container.query_items.assert_called_once()


@pytest.mark.asyncio
async def test_list_with_prefix(cosmos_adapter):
    """Test listing keys with prefix filter."""
    mock_items = [{"id": "agent-1:conv"}, {"id": "agent-1:ctx"}]

    async def mock_iter():
        for item in mock_items:
            yield item

    mock_container = MagicMock()
    mock_container.query_items.return_value = mock_iter()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        keys = await cosmos_adapter.alist(prefix="agent-1")

        assert len(keys) == 2
        call_args = mock_container.query_items.call_args
        assert "STARTSWITH" in call_args[1]["query"]


@pytest.mark.asyncio
async def test_store_conversation(cosmos_adapter):
    """Test storing conversation history."""
    mock_container = AsyncMock()
    mock_container.upsert_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        await cosmos_adapter.astore_conversation("agent-1", messages, "session-1")

        call_args = mock_container.upsert_item.call_args
        document = call_args[0][0]
        assert document["id"] == "agent-1:conversation:session-1"
        assert document["data"]["messages"] == messages


@pytest.mark.asyncio
async def test_retrieve_conversation(cosmos_adapter):
    """Test retrieving conversation history."""
    mock_item = {
        "id": "agent-1:conversation:session-1",
        "data": {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        },
    }

    mock_container = AsyncMock()
    mock_container.read_item = AsyncMock(return_value=mock_item)

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        messages = await cosmos_adapter.aretrieve_conversation("agent-1", "session-1")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_retrieve_conversation_not_found(cosmos_adapter):
    """Test retrieving missing conversation returns empty list."""
    mock_container = AsyncMock()
    mock_container.read_item = AsyncMock(side_effect=Exception("NotFound"))

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        messages = await cosmos_adapter.aretrieve_conversation("agent-1", "missing")

        assert messages == []


@pytest.mark.asyncio
async def test_clear_agent(cosmos_adapter):
    """Test clearing all agent data."""
    mock_items = [{"id": "agent-1:conv"}, {"id": "agent-1:ctx"}]

    async def mock_iter():
        for item in mock_items:
            yield item

    mock_container = MagicMock()
    mock_container.query_items.return_value = mock_iter()
    mock_container.delete_item = AsyncMock()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        count = await cosmos_adapter.aclear_agent("agent-1")

        assert count == 2


@pytest.mark.asyncio
async def test_search(cosmos_adapter):
    """Test searching memories."""
    mock_items = [
        {"id": "key-1", "data": {"content": "test query"}, "metadata": {}, "created_at": 1234567890}
    ]

    async def mock_iter():
        for item in mock_items:
            yield item

    mock_container = MagicMock()
    mock_container.query_items.return_value = mock_iter()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        results = await cosmos_adapter.asearch("test", top_k=5)

        assert len(results) == 1
        call_args = mock_container.query_items.call_args
        assert "CONTAINS" in call_args[1]["query"]


@pytest.mark.asyncio
async def test_health_check_healthy(cosmos_adapter):
    """Test health check returns healthy status."""

    async def mock_iter():
        yield 42  # Document count

    mock_container = MagicMock()
    mock_container.query_items.return_value = mock_iter()

    with patch.object(cosmos_adapter, "_get_container", return_value=mock_container):
        status = await cosmos_adapter.ahealth_check()

        assert status.status == "healthy"
        assert status.adapter_name == "AzureCosmos"
        assert status.details["database"] == "test-db"


@pytest.mark.asyncio
async def test_health_check_unhealthy(cosmos_adapter):
    """Test health check returns unhealthy on error."""
    with patch.object(cosmos_adapter, "_get_container", side_effect=Exception("Connection failed")):
        status = await cosmos_adapter.ahealth_check()

        assert status.status == "unhealthy"
        assert "error" in status.details


@pytest.mark.asyncio
async def test_to_dict(cosmos_adapter):
    """Test serialization excludes secrets."""
    config = cosmos_adapter.to_dict()

    assert "endpoint" in config
    assert "database_name" in config
    assert "key" not in config  # Secret excluded


@pytest.mark.asyncio
async def test_key_required_without_managed_identity():
    """Test key is required without managed identity."""
    adapter = AzureCosmosAdapter(
        endpoint="https://test.documents.azure.com:443/",
        use_managed_identity=False,
        database_name="test",
    )

    with pytest.raises(ValueError, match="key is required"):
        await adapter._get_container()
