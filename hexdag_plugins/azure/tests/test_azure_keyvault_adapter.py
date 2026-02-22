"""Tests for Azure Key Vault adapter."""

import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hexdag.kernel.types import Secret

from hexdag_plugins.azure.adapters.keyvault import AzureKeyVaultAdapter


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


# ===================================================================
# Initialization
# ===================================================================


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
async def test_service_principal_requires_all_credentials():
    """Test service principal auth requires all credentials."""
    adapter = AzureKeyVaultAdapter(
        vault_url="https://test.vault.azure.net",
        use_managed_identity=False,
        tenant_id="tenant",
        # Missing client_id and client_secret
    )

    # Mock the Azure SDK import so we only test the validation logic
    mock_azure_identity = MagicMock()
    mock_azure_keyvault = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {
                "azure": MagicMock(),
                "azure.identity": mock_azure_identity,
                "azure.keyvault": MagicMock(),
                "azure.keyvault.secrets": mock_azure_keyvault,
            },
        ),
        pytest.raises(ValueError, match="tenant_id, client_id, and client_secret"),
    ):
        adapter._get_client()


# ===================================================================
# aget_secret() — SecretPort protocol
# ===================================================================


@pytest.mark.asyncio
async def test_aget_secret_returns_secret_wrapper(keyvault_adapter):
    """Test aget_secret() returns Secret wrapper."""
    mock_secret = MagicMock()
    mock_secret.value = "secret-value-123"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget_secret("MY-SECRET")

        assert isinstance(result, Secret)
        assert result.get() == "secret-value-123"
        assert str(result) == "<SECRET>"
        mock_client.get_secret.assert_called_once_with("MY-SECRET")


@pytest.mark.asyncio
async def test_aget_secret_from_cache(keyvault_adapter):
    """Test secret retrieval from cache."""
    keyvault_adapter._cache["CACHED-SECRET"] = ("cached-value", time.time())

    mock_client = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget_secret("CACHED-SECRET")

        assert isinstance(result, Secret)
        assert result.get() == "cached-value"
        mock_client.get_secret.assert_not_called()


@pytest.mark.asyncio
async def test_aget_secret_expired_cache(keyvault_adapter):
    """Test secret retrieval with expired cache."""
    keyvault_adapter._cache["EXPIRED-SECRET"] = ("old-value", time.time() - 400)

    mock_secret = MagicMock()
    mock_secret.value = "new-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        result = await keyvault_adapter.aget_secret("EXPIRED-SECRET")

        assert result.get() == "new-value"
        mock_client.get_secret.assert_called_once()


@pytest.mark.asyncio
async def test_aget_secret_no_cache(keyvault_adapter_no_cache):
    """Test secret retrieval with caching disabled."""
    mock_secret = MagicMock()
    mock_secret.value = "secret-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter_no_cache, "_get_client", return_value=mock_client):
        await keyvault_adapter_no_cache.aget_secret("NO-CACHE-SECRET")
        await keyvault_adapter_no_cache.aget_secret("NO-CACHE-SECRET")

        assert mock_client.get_secret.call_count == 2


@pytest.mark.asyncio
async def test_aget_secret_raises_key_error_for_missing(keyvault_adapter):
    """Test aget_secret() raises KeyError when secret not found."""
    mock_client = MagicMock()
    mock_client.get_secret.side_effect = Exception("SecretNotFound: Secret not found")

    with (
        patch.object(keyvault_adapter, "_get_client", return_value=mock_client),
        pytest.raises(KeyError, match="not found"),
    ):
        await keyvault_adapter.aget_secret("MISSING-SECRET")


@pytest.mark.asyncio
async def test_aget_secret_null_value(keyvault_adapter):
    """Test aget_secret() raises ValueError for null value."""
    mock_secret = MagicMock()
    mock_secret.value = None

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with (
        patch.object(keyvault_adapter, "_get_client", return_value=mock_client),
        pytest.raises(ValueError, match="has no value"),
    ):
        await keyvault_adapter.aget_secret("NULL-SECRET")


# ===================================================================
# aload_secrets_to_memory() — SecretPort protocol
# ===================================================================


@pytest.mark.asyncio
async def test_aload_secrets_to_memory(keyvault_adapter):
    """Test aload_secrets_to_memory() stores secrets in Memory with prefix."""
    mock_secret = MagicMock()
    mock_secret.value = "secret-value-123"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    mock_memory = AsyncMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        mapping = await keyvault_adapter.aload_secrets_to_memory(
            memory=mock_memory,
            prefix="secret:",
            keys=["MY-SECRET", "OTHER-SECRET"],
        )

        assert mapping == {
            "MY-SECRET": "secret:MY-SECRET",
            "OTHER-SECRET": "secret:OTHER-SECRET",
        }
        assert mock_memory.aset.call_count == 2


@pytest.mark.asyncio
async def test_aload_secrets_to_memory_auto_discover(keyvault_adapter):
    """Test aload_secrets_to_memory() auto-discovers keys when None."""
    mock_prop1 = MagicMock()
    mock_prop1.name = "SECRET-A"
    mock_prop2 = MagicMock()
    mock_prop2.name = "SECRET-B"

    mock_secret = MagicMock()
    mock_secret.value = "val"

    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = [mock_prop1, mock_prop2]
    mock_client.get_secret.return_value = mock_secret

    mock_memory = AsyncMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        mapping = await keyvault_adapter.aload_secrets_to_memory(memory=mock_memory, keys=None)

        assert len(mapping) == 2
        assert "SECRET-A" in mapping
        assert "SECRET-B" in mapping


@pytest.mark.asyncio
async def test_aload_secrets_to_memory_skips_missing(keyvault_adapter):
    """Test aload_secrets_to_memory() skips missing secrets gracefully."""

    def mock_get_secret(name):
        if name == "MISSING":
            raise Exception("SecretNotFound")
        mock = MagicMock()
        mock.value = "found-value"
        return mock

    mock_client = MagicMock()
    mock_client.get_secret.side_effect = mock_get_secret

    mock_memory = AsyncMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        mapping = await keyvault_adapter.aload_secrets_to_memory(
            memory=mock_memory, keys=["GOOD-SECRET", "MISSING"]
        )

        assert len(mapping) == 1
        assert "GOOD-SECRET" in mapping
        assert "MISSING" not in mapping


# ===================================================================
# alist_secret_names() — SecretPort protocol
# ===================================================================


@pytest.mark.asyncio
async def test_alist_secret_names(keyvault_adapter):
    """Test listing all secret names."""
    mock_secret1 = MagicMock()
    mock_secret1.name = "SECRET1"
    mock_secret2 = MagicMock()
    mock_secret2.name = "SECRET2"

    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = [mock_secret1, mock_secret2]

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        names = await keyvault_adapter.alist_secret_names()

        assert names == ["SECRET1", "SECRET2"]


# ===================================================================
# Additional operations (aset, adelete, clear_cache)
# ===================================================================


@pytest.mark.asyncio
async def test_set_secret(keyvault_adapter):
    """Test setting a secret."""
    mock_client = MagicMock()
    mock_client.set_secret = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        await keyvault_adapter.aset("NEW-SECRET", "secret-value")

        mock_client.set_secret.assert_called_once_with("NEW-SECRET", "secret-value")
        assert "NEW-SECRET" in keyvault_adapter._cache


@pytest.mark.asyncio
async def test_delete_secret(keyvault_adapter):
    """Test deleting a secret."""
    keyvault_adapter._cache["TO-DELETE"] = ("value", time.time())

    mock_client = MagicMock()
    mock_client.begin_delete_secret = MagicMock()

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        await keyvault_adapter.adelete("TO-DELETE")

        mock_client.begin_delete_secret.assert_called_once_with("TO-DELETE")
        assert "TO-DELETE" not in keyvault_adapter._cache


@pytest.mark.asyncio
async def test_clear_cache(keyvault_adapter):
    """Test clearing the cache."""
    keyvault_adapter._cache["SECRET1"] = ("value1", time.time())
    keyvault_adapter._cache["SECRET2"] = ("value2", time.time())

    keyvault_adapter.clear_cache()

    assert len(keyvault_adapter._cache) == 0


# ===================================================================
# Health check
# ===================================================================


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


# ===================================================================
# Serialization
# ===================================================================


@pytest.mark.asyncio
async def test_to_dict(keyvault_adapter):
    """Test serialization excludes credentials."""
    config = keyvault_adapter.to_dict()

    assert "vault_url" in config
    assert "cache_secrets" in config
    assert "client_secret" not in config


# ===================================================================
# load_to_environ()
# ===================================================================


@pytest.mark.asyncio
async def test_load_to_environ_sets_env_vars(keyvault_adapter):
    """Test load_to_environ() sets os.environ with hyphen-to-underscore normalization."""
    mock_secret = MagicMock()
    mock_secret.value = "sk-test-key-123"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    os.environ.pop("OPENAI_API_KEY", None)

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        try:
            results = await keyvault_adapter.load_to_environ(keys=["OPENAI-API-KEY"])

            assert results["OPENAI_API_KEY"] == "loaded"
            assert os.environ["OPENAI_API_KEY"] == "sk-test-key-123"
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


@pytest.mark.asyncio
async def test_load_to_environ_skips_existing(keyvault_adapter):
    """Test load_to_environ() skips existing env vars when overwrite=False."""
    os.environ["EXISTING_KEY"] = "original-value"

    mock_secret = MagicMock()
    mock_secret.value = "new-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        try:
            results = await keyvault_adapter.load_to_environ(keys=["EXISTING-KEY"], overwrite=False)

            assert results["EXISTING_KEY"] == "skipped"
            assert os.environ["EXISTING_KEY"] == "original-value"
        finally:
            del os.environ["EXISTING_KEY"]


@pytest.mark.asyncio
async def test_load_to_environ_overwrites_existing(keyvault_adapter):
    """Test load_to_environ() overwrites existing env vars when overwrite=True."""
    os.environ["OVERWRITE_KEY"] = "original-value"

    mock_secret = MagicMock()
    mock_secret.value = "new-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        try:
            results = await keyvault_adapter.load_to_environ(keys=["OVERWRITE-KEY"], overwrite=True)

            assert results["OVERWRITE_KEY"] == "loaded"
            assert os.environ["OVERWRITE_KEY"] == "new-value"
        finally:
            del os.environ["OVERWRITE_KEY"]


@pytest.mark.asyncio
async def test_load_to_environ_with_prefix(keyvault_adapter):
    """Test load_to_environ() applies prefix to env var names."""
    mock_secret = MagicMock()
    mock_secret.value = "prefixed-value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    os.environ.pop("MYAPP_DB_PASSWORD", None)

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        try:
            results = await keyvault_adapter.load_to_environ(keys=["DB-PASSWORD"], prefix="MYAPP_")

            assert results["MYAPP_DB_PASSWORD"] == "loaded"
            assert os.environ["MYAPP_DB_PASSWORD"] == "prefixed-value"
        finally:
            os.environ.pop("MYAPP_DB_PASSWORD", None)


@pytest.mark.asyncio
async def test_load_to_environ_auto_discover(keyvault_adapter):
    """Test load_to_environ() auto-discovers keys when None."""
    mock_prop1 = MagicMock()
    mock_prop1.name = "AUTO-KEY-1"
    mock_prop2 = MagicMock()
    mock_prop2.name = "AUTO-KEY-2"

    mock_secret = MagicMock()
    mock_secret.value = "auto-val"

    mock_client = MagicMock()
    mock_client.list_properties_of_secrets.return_value = [mock_prop1, mock_prop2]
    mock_client.get_secret.return_value = mock_secret

    os.environ.pop("AUTO_KEY_1", None)
    os.environ.pop("AUTO_KEY_2", None)

    with patch.object(keyvault_adapter, "_get_client", return_value=mock_client):
        try:
            results = await keyvault_adapter.load_to_environ(keys=None)

            assert results["AUTO_KEY_1"] == "loaded"
            assert results["AUTO_KEY_2"] == "loaded"
            assert os.environ["AUTO_KEY_1"] == "auto-val"
            assert os.environ["AUTO_KEY_2"] == "auto-val"
        finally:
            os.environ.pop("AUTO_KEY_1", None)
            os.environ.pop("AUTO_KEY_2", None)
