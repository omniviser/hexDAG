"""Tests for Azure Key Vault adapter."""

import time
from unittest.mock import MagicMock, patch

import pytest

from hexdag_plugins.azure.azure_keyvault_adapter import AzureKeyVaultAdapter


@pytest.fixture
def keyvault_adapter():
    """Create Azure Key Vault adapter for testing."""
    return AzureKeyVaultAdapter(
        vault_url="https://test-vault.vault.azure.net",
        use_managed_identity=True,
    )


@pytest.fixture
def keyvault_adapter_service_principal():
    """Create Azure Key Vault adapter with service principal auth."""
    return AzureKeyVaultAdapter(
        vault_url="https://test-vault.vault.azure.net",
        use_managed_identity=False,
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
    )


@pytest.fixture
def keyvault_adapter_no_cache():
    """Create adapter with caching disabled."""
    return AzureKeyVaultAdapter(
        vault_url="https://test-vault.vault.azure.net",
        use_managed_identity=True,
        cache_secrets=False,
    )


@pytest.mark.asyncio
async def test_adapter_initialization(keyvault_adapter):
    """Test adapter initializes with correct parameters."""
    assert keyvault_adapter.vault_url == "https://test-vault.vault.azure.net"
    assert keyvault_adapter.use_managed_identity is True
    assert keyvault_adapter.cache_secrets is True
    assert keyvault_adapter.cache_ttl == 300
    assert keyvault_adapter._client is None


@pytest.mark.asyncio
async def test_adapter_initialization_service_principal(keyvault_adapter_service_principal):
    """Test adapter initializes with service principal."""
    assert keyvault_adapter_service_principal.use_managed_identity is False
    assert keyvault_adapter_service_principal.tenant_id == "test-tenant-id"
    assert keyvault_adapter_service_principal.client_id == "test-client-id"


@pytest.mark.asyncio
async def test_get_secret_success(keyvault_adapter):
    """Test successful secret retrieval."""
    mock_secret = MagicMock()
    mock_secret.value = "secret-value-123"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget("MY-SECRET")

        assert result == "secret-value-123"
        mock_client.get_secret.assert_called_once_with("MY-SECRET")


@pytest.mark.asyncio
async def test_get_secret_from_cache(keyvault_adapter):
    """Test secret retrieval from cache."""
    # Pre-populate cache
    keyvault_adapter._cache["CACHED-SECRET"] = ("cached-value", time.time())

    mock_client = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget("CACHED-SECRET")

        assert result == "cached-value"
        mock_client.get_secret.assert_not_called()  # Should not hit the API


@pytest.mark.asyncio
async def test_get_secret_expired_cache(keyvault_adapter):
    """Test secret retrieval with expired cache."""
    # Pre-populate cache with expired entry
    keyvault_adapter._cache["EXPIRED-SECRET"] = ("old-value", time.time() - 400)

    mock_secret = MagicMock()
    mock_secret.value = "new-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget("EXPIRED-SECRET")

        assert result == "new-value"
        mock_client.get_secret.assert_called_once()


@pytest.mark.asyncio
async def test_get_secret_no_cache(keyvault_adapter_no_cache):
    """Test secret retrieval with caching disabled."""
    mock_secret = MagicMock()
    mock_secret.value = "secret-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter_no_cache, "_get_client", return_value=mock_client):
        # First call
        await keyvault_adapter_no_cache.aget("NO-CACHE-SECRET")
        # Second call should also hit the API
        await keyvault_adapter_no_cache.aget("NO-CACHE-SECRET")

        assert mock_client.get_secret.call_count == 2


@pytest.mark.asyncio
async def test_get_secret_not_found(keyvault_adapter):
    """Test secret retrieval raises ValueError for missing secret."""
    mock_client = MagicMock()
    mock_client.get_secret.side_effect = Exception("SecretNotFound: Secret not found")

    with (
        patch.object(keyvault_adapter, "_get_client", return_value=mock_client),
        pytest.raises(ValueError, match="not found"),
    ):
        await keyvault_adapter.aget("MISSING-SECRET")


@pytest.mark.asyncio
async def test_get_secret_null_value(keyvault_adapter):
    """Test secret retrieval raises ValueError for null value."""
    mock_secret = MagicMock()
    mock_secret.value = None

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with (
        patch.object(keyvault_adapter, "_get_client", return_value=mock_client),
        pytest.raises(ValueError, match="has no value"),
    ):
        await keyvault_adapter.aget("NULL-SECRET")


@pytest.mark.asyncio
async def test_get_batch_success(keyvault_adapter):
    """Test batch secret retrieval."""

    def mock_get_secret(name):
        mock = MagicMock()
        mock.value = f"value-for-{name}"
        return mock

    mock_client = MagicMock()
    mock_client.get_secret.side_effect = mock_get_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        results = await keyvault_adapter.aget_batch(["SECRET1", "SECRET2"])

        assert len(results) == 2
        assert results["SECRET1"] == "value-for-SECRET1"
        assert results["SECRET2"] == "value-for-SECRET2"


@pytest.mark.asyncio
async def test_get_batch_partial_success(keyvault_adapter):
    """Test batch retrieval skips missing secrets."""

    def mock_get_secret(name):
        if name == "MISSING":
            raise Exception("SecretNotFound")
        mock = MagicMock()
        mock.value = f"value-for-{name}"
        return mock

    mock_client = MagicMock()
    mock_client.get_secret.side_effect = mock_get_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        results = await keyvault_adapter.aget_batch(["SECRET1", "MISSING", "SECRET2"])

        assert len(results) == 2
        assert "SECRET1" in results
        assert "MISSING" not in results
        assert "SECRET2" in results


@pytest.mark.asyncio
async def test_set_secret(keyvault_adapter):
    """Test setting a secret."""
    mock_client = MagicMock()
    mock_client.set_secret = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        await keyvault_adapter.aset("NEW-SECRET", "secret-value")

        mock_client.set_secret.assert_called_once_with("NEW-SECRET", "secret-value")
        # Verify it's also cached
        assert "NEW-SECRET" in keyvault_adapter._cache


@pytest.mark.asyncio
async def test_delete_secret(keyvault_adapter):
    """Test deleting a secret."""
    # Pre-populate cache
    keyvault_adapter._cache["TO-DELETE"] = ("value", time.time())

    mock_client = MagicMock()
    mock_client.begin_delete_secret = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        await keyvault_adapter.adelete("TO-DELETE")

        mock_client.begin_delete_secret.assert_called_once_with("TO-DELETE")
        assert "TO-DELETE" not in keyvault_adapter._cache


@pytest.mark.asyncio
async def test_list_secrets(keyvault_adapter):
    """Test listing all secrets."""
    mock_secret1 = MagicMock()
    mock_secret1.name = "SECRET1"
    mock_secret2 = MagicMock()
    mock_secret2.name = "SECRET2"

    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = [mock_secret1, mock_secret2]

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        names = await keyvault_adapter.alist()

        assert names == ["SECRET1", "SECRET2"]


@pytest.mark.asyncio
async def test_clear_cache(keyvault_adapter):
    """Test clearing the cache."""
    keyvault_adapter._cache["SECRET1"] = ("value1", time.time())
    keyvault_adapter._cache["SECRET2"] = ("value2", time.time())

    keyvault_adapter.clear_cache()

    assert len(keyvault_adapter._cache) == 0


@pytest.mark.asyncio
async def test_health_check_healthy(keyvault_adapter):
    """Test health check returns healthy status."""
    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = []

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        status = await keyvault_adapter.ahealth_check()

        assert status.status == "healthy"
        assert status.adapter_name == "AzureKeyVault"
        assert status.details["vault_url"] == "https://test-vault.vault.azure.net"
        assert status.details["auth_method"] == "managed_identity"


@pytest.mark.asyncio
async def test_health_check_service_principal(keyvault_adapter_service_principal):
    """Test health check shows service principal auth method."""
    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = []

    with patch.object(keyvault_adapter_service_principal, "_get_client", return_value=mock_client):
        status = await keyvault_adapter_service_principal.ahealth_check()

        assert status.details["auth_method"] == "service_principal"


@pytest.mark.asyncio
async def test_health_check_unhealthy(keyvault_adapter):
    """Test health check returns unhealthy on error."""
    with patch.object(keyvault_adapter, "_get_client", side_effect=Exception("Connection failed")):
        status = await keyvault_adapter.ahealth_check()

        assert status.status == "unhealthy"
        assert "error" in status.details


@pytest.mark.asyncio
async def test_to_dict(keyvault_adapter):
    """Test serialization excludes credentials."""
    config = keyvault_adapter.to_dict()

    assert "vault_url" in config
    assert "cache_secrets" in config
    assert "client_secret" not in config  # Credentials excluded


@pytest.mark.asyncio
async def test_service_principal_requires_all_credentials():
    """Test service principal auth requires all credentials."""
    adapter = AzureKeyVaultAdapter(
        vault_url="https://test.vault.azure.net",
        use_managed_identity=False,
        tenant_id="tenant",
        # Missing client_id and client_secret
    )

    with pytest.raises(ValueError, match="tenant_id, client_id, and client_secret"):
        adapter._get_client()


@pytest.mark.asyncio
async def test_client_lazy_initialization(keyvault_adapter):
    """Test client is lazily initialized."""
    assert keyvault_adapter._client is None

    # The _get_client method creates the client
    with (
        patch("hexdag_plugins.azure.azure_keyvault_adapter.DefaultAzureCredential"),
        patch("hexdag_plugins.azure.azure_keyvault_adapter.SecretClient") as mock_client,
    ):
        keyvault_adapter._get_client()
        mock_client.assert_called_once()
