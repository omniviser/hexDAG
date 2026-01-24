"""Tests for Azure Blob Storage adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag_plugins.azure.azure_blob_adapter import AzureBlobAdapter


@pytest.fixture
def blob_adapter():
    """Create Azure Blob adapter for testing."""
    return AzureBlobAdapter(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=testkey123;EndpointSuffix=core.windows.net",
        container_name="test-container",
    )


@pytest.fixture
def blob_adapter_managed_identity():
    """Create Azure Blob adapter with managed identity."""
    return AzureBlobAdapter(
        account_url="https://teststorage.blob.core.windows.net",
        container_name="test-container",
        use_managed_identity=True,
    )


@pytest.mark.asyncio
async def test_adapter_initialization(blob_adapter):
    """Test adapter initializes with correct parameters."""
    assert blob_adapter.container_name == "test-container"
    assert blob_adapter.use_managed_identity is False
    assert blob_adapter._container_client is None


@pytest.mark.asyncio
async def test_adapter_initialization_managed_identity(blob_adapter_managed_identity):
    """Test adapter initializes with managed identity."""
    assert blob_adapter_managed_identity.account_url == "https://teststorage.blob.core.windows.net"
    assert blob_adapter_managed_identity.use_managed_identity is True


@pytest.mark.asyncio
async def test_upload_success(blob_adapter):
    """Test successful blob upload."""
    mock_blob_client = AsyncMock()
    mock_blob_client.url = "https://test.blob.core.windows.net/test-container/test.txt"
    mock_blob_client.upload_blob = AsyncMock()

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        url = await blob_adapter.aupload("test.txt", b"hello world")

        assert url == "https://test.blob.core.windows.net/test-container/test.txt"
        mock_blob_client.upload_blob.assert_called_once()


@pytest.mark.asyncio
async def test_upload_string(blob_adapter):
    """Test uploading string data."""
    mock_blob_client = AsyncMock()
    mock_blob_client.url = "https://test.blob.core.windows.net/test-container/test.txt"
    mock_blob_client.upload_blob = AsyncMock()

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        await blob_adapter.aupload("test.txt", "hello string")

        call_args = mock_blob_client.upload_blob.call_args
        assert call_args.kwargs.get("content_type") == "text/plain"


@pytest.mark.asyncio
async def test_download_success(blob_adapter):
    """Test successful blob download."""
    mock_stream = AsyncMock()
    mock_stream.readall = AsyncMock(return_value=b"downloaded content")

    mock_blob_client = AsyncMock()
    mock_blob_client.download_blob = AsyncMock(return_value=mock_stream)

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        content = await blob_adapter.adownload("test.txt")

        assert content == b"downloaded content"


@pytest.mark.asyncio
async def test_download_not_found(blob_adapter):
    """Test download raises FileNotFoundError for missing blob."""
    mock_blob_client = AsyncMock()
    mock_blob_client.download_blob = AsyncMock(
        side_effect=Exception("BlobNotFound: The specified blob does not exist")
    )

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with (
        patch.object(blob_adapter, "_get_container", return_value=mock_container),
        pytest.raises(FileNotFoundError, match="not found"),
    ):
        await blob_adapter.adownload("missing.txt")


@pytest.mark.asyncio
async def test_download_text(blob_adapter):
    """Test downloading blob as text."""
    mock_stream = AsyncMock()
    mock_stream.readall = AsyncMock(return_value=b"text content")

    mock_blob_client = AsyncMock()
    mock_blob_client.download_blob = AsyncMock(return_value=mock_stream)

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        text = await blob_adapter.adownload_text("test.txt")

        assert text == "text content"


@pytest.mark.asyncio
async def test_delete_success(blob_adapter):
    """Test successful blob deletion."""
    mock_blob_client = AsyncMock()
    mock_blob_client.delete_blob = AsyncMock()

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        result = await blob_adapter.adelete("test.txt")

        assert result is True
        mock_blob_client.delete_blob.assert_called_once()


@pytest.mark.asyncio
async def test_delete_not_found(blob_adapter):
    """Test delete returns False for missing blob."""
    mock_blob_client = AsyncMock()
    mock_blob_client.delete_blob = AsyncMock(side_effect=Exception("BlobNotFound"))

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        result = await blob_adapter.adelete("missing.txt")

        assert result is False


@pytest.mark.asyncio
async def test_exists_true(blob_adapter):
    """Test exists returns True for existing blob."""
    mock_blob_client = AsyncMock()
    mock_blob_client.exists = AsyncMock(return_value=True)

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        result = await blob_adapter.aexists("test.txt")

        assert result is True


@pytest.mark.asyncio
async def test_exists_false(blob_adapter):
    """Test exists returns False for missing blob."""
    mock_blob_client = AsyncMock()
    mock_blob_client.exists = AsyncMock(return_value=False)

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        result = await blob_adapter.aexists("missing.txt")

        assert result is False


@pytest.mark.asyncio
async def test_list_blobs(blob_adapter):
    """Test listing blobs."""
    # Create mock blobs
    mock_blob1 = MagicMock()
    mock_blob1.name = "file1.txt"
    mock_blob1.size = 100
    mock_blob1.last_modified = "2024-01-01"
    mock_blob1.content_settings = MagicMock()
    mock_blob1.content_settings.content_type = "text/plain"
    mock_blob1.metadata = {"key": "value"}

    mock_blob2 = MagicMock()
    mock_blob2.name = "file2.txt"
    mock_blob2.size = 200
    mock_blob2.last_modified = "2024-01-02"
    mock_blob2.content_settings = None
    mock_blob2.metadata = None

    # Create async iterator
    async def mock_list_blobs(name_starts_with=None):
        for blob in [mock_blob1, mock_blob2]:
            yield blob

    mock_container = AsyncMock()
    mock_container.list_blobs = mock_list_blobs

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        results = await blob_adapter.alist()

        assert len(results) == 2
        assert results[0]["name"] == "file1.txt"
        assert results[0]["size"] == 100
        assert results[1]["name"] == "file2.txt"


@pytest.mark.asyncio
async def test_list_blobs_with_prefix(blob_adapter):
    """Test listing blobs with prefix filter."""
    mock_blob = MagicMock()
    mock_blob.name = "reports/file1.txt"
    mock_blob.size = 100
    mock_blob.last_modified = "2024-01-01"
    mock_blob.content_settings = None
    mock_blob.metadata = None

    async def mock_list_blobs(name_starts_with=None):
        if name_starts_with == "reports/":
            yield mock_blob

    mock_container = AsyncMock()
    mock_container.list_blobs = mock_list_blobs

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        results = await blob_adapter.alist(prefix="reports/")

        assert len(results) == 1
        assert results[0]["name"] == "reports/file1.txt"


@pytest.mark.asyncio
async def test_upload_json(blob_adapter):
    """Test uploading JSON data."""
    mock_blob_client = AsyncMock()
    mock_blob_client.url = "https://test.blob.core.windows.net/test-container/data.json"
    mock_blob_client.upload_blob = AsyncMock()

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        url = await blob_adapter.aupload_json("data.json", {"key": "value"})

        assert url == "https://test.blob.core.windows.net/test-container/data.json"
        call_args = mock_blob_client.upload_blob.call_args
        assert call_args.kwargs.get("content_type") == "application/json"


@pytest.mark.asyncio
async def test_download_json(blob_adapter):
    """Test downloading and parsing JSON."""
    mock_stream = AsyncMock()
    mock_stream.readall = AsyncMock(return_value=b'{"key": "value"}')

    mock_blob_client = AsyncMock()
    mock_blob_client.download_blob = AsyncMock(return_value=mock_stream)

    mock_container = AsyncMock()
    mock_container.get_blob_client.return_value = mock_blob_client

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        data = await blob_adapter.adownload_json("data.json")

        assert data == {"key": "value"}


@pytest.mark.asyncio
async def test_health_check_healthy(blob_adapter):
    """Test health check returns healthy status."""
    mock_blob = MagicMock()
    mock_blob.name = "file.txt"

    async def mock_list_blobs():
        yield mock_blob

    mock_container = AsyncMock()
    mock_container.list_blobs = mock_list_blobs

    with patch.object(blob_adapter, "_get_container", return_value=mock_container):
        status = await blob_adapter.ahealth_check()

        assert status.status == "healthy"
        assert status.adapter_name == "AzureBlob"
        assert status.details["container"] == "test-container"


@pytest.mark.asyncio
async def test_health_check_unhealthy(blob_adapter):
    """Test health check returns unhealthy on error."""
    with patch.object(blob_adapter, "_get_container", side_effect=Exception("Connection failed")):
        status = await blob_adapter.ahealth_check()

        assert status.status == "unhealthy"
        assert "error" in status.details


@pytest.mark.asyncio
async def test_to_dict(blob_adapter):
    """Test serialization excludes secrets."""
    config = blob_adapter.to_dict()

    assert "container_name" in config
    assert config["container_name"] == "test-container"
    assert "connection_string" not in config  # Secret excluded


@pytest.mark.asyncio
async def test_managed_identity_requires_account_url():
    """Test managed identity requires account_url."""
    adapter = AzureBlobAdapter(
        use_managed_identity=True,
        container_name="test",
    )

    with pytest.raises(ValueError, match="account_url is required"):
        await adapter._get_container()


@pytest.mark.asyncio
async def test_connection_string_required_without_managed_identity():
    """Test connection_string required without managed identity."""
    adapter = AzureBlobAdapter(
        use_managed_identity=False,
        container_name="test",
    )

    with pytest.raises(ValueError, match="connection_string is required"):
        await adapter._get_container()
