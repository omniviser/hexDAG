"""Tests for the LocalSecretAdapter module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hexdag.builtin.adapters.secret.local_secret_adapter import LocalSecretAdapter
from hexdag.core.types import Secret


class TestLocalSecretAdapterInit:
    """Tests for LocalSecretAdapter initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with no arguments."""
        adapter = LocalSecretAdapter()
        assert adapter.env_prefix == ""
        assert adapter.allow_empty is False
        assert adapter._cache == {}

    def test_initialization_with_env_prefix(self) -> None:
        """Test initialization with env_prefix."""
        adapter = LocalSecretAdapter(env_prefix="MYAPP_")
        assert adapter.env_prefix == "MYAPP_"
        assert adapter.allow_empty is False

    def test_initialization_with_allow_empty(self) -> None:
        """Test initialization with allow_empty=True."""
        adapter = LocalSecretAdapter(allow_empty=True)
        assert adapter.env_prefix == ""
        assert adapter.allow_empty is True

    def test_initialization_with_all_options(self) -> None:
        """Test initialization with all options."""
        adapter = LocalSecretAdapter(env_prefix="TEST_", allow_empty=True)
        assert adapter.env_prefix == "TEST_"
        assert adapter.allow_empty is True


class TestAgetSecret:
    """Tests for aget_secret method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up test environment variables."""
        monkeypatch.setenv("TEST_API_KEY", "secret-value-123")
        monkeypatch.setenv("MYAPP_DB_PASSWORD", "db-password")
        monkeypatch.setenv("EMPTY_SECRET", "")

    @pytest.mark.asyncio
    async def test_get_secret_success(self) -> None:
        """Test successfully retrieving a secret."""
        adapter = LocalSecretAdapter()
        secret = await adapter.aget_secret("TEST_API_KEY")
        assert isinstance(secret, Secret)
        assert secret.get() == "secret-value-123"

    @pytest.mark.asyncio
    async def test_get_secret_with_prefix(self) -> None:
        """Test retrieving a secret with env prefix."""
        adapter = LocalSecretAdapter(env_prefix="MYAPP_")
        secret = await adapter.aget_secret("DB_PASSWORD")
        assert secret.get() == "db-password"

    @pytest.mark.asyncio
    async def test_get_secret_not_found(self) -> None:
        """Test KeyError when secret not found."""
        adapter = LocalSecretAdapter()
        with pytest.raises(KeyError) as exc_info:
            await adapter.aget_secret("NONEXISTENT_KEY")
        assert "NONEXISTENT_KEY" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_secret_empty_not_allowed(self) -> None:
        """Test ValueError when empty secret not allowed."""
        adapter = LocalSecretAdapter(allow_empty=False)
        with pytest.raises(ValueError) as exc_info:
            await adapter.aget_secret("EMPTY_SECRET")
        assert "cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_secret_empty_allowed(self) -> None:
        """Test empty secret is returned when allow_empty=True."""
        adapter = LocalSecretAdapter(allow_empty=True)
        secret = await adapter.aget_secret("EMPTY_SECRET")
        assert secret.get() == ""

    @pytest.mark.asyncio
    async def test_get_secret_cached(self) -> None:
        """Test that secrets are cached."""
        adapter = LocalSecretAdapter()
        secret1 = await adapter.aget_secret("TEST_API_KEY")
        secret2 = await adapter.aget_secret("TEST_API_KEY")
        assert secret1 is secret2
        assert "TEST_API_KEY" in adapter._cache

    @pytest.mark.asyncio
    async def test_get_secret_with_prefix_not_found(self) -> None:
        """Test KeyError includes full env var name."""
        adapter = LocalSecretAdapter(env_prefix="MISSING_")
        with pytest.raises(KeyError) as exc_info:
            await adapter.aget_secret("KEY")
        assert "MISSING_KEY" in str(exc_info.value)


class TestAloadSecretsToMemory:
    """Tests for aload_secrets_to_memory method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up test environment variables."""
        monkeypatch.setenv("SECRET_API_KEY", "api-key-value")
        monkeypatch.setenv("SECRET_DB_PASS", "db-pass-value")
        monkeypatch.setenv("OTHER_VAR", "other-value")

    @pytest.fixture
    def mock_memory(self) -> AsyncMock:
        """Create a mock memory port."""
        memory = AsyncMock()
        memory.aset = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_load_specific_secrets(self, mock_memory: AsyncMock) -> None:
        """Test loading specific secrets to memory."""
        adapter = LocalSecretAdapter()
        mapping = await adapter.aload_secrets_to_memory(
            memory=mock_memory,
            keys=["SECRET_API_KEY"],
        )
        assert mapping == {"SECRET_API_KEY": "secret:SECRET_API_KEY"}
        mock_memory.aset.assert_called_once_with("secret:SECRET_API_KEY", "api-key-value")

    @pytest.mark.asyncio
    async def test_load_multiple_secrets(self, mock_memory: AsyncMock) -> None:
        """Test loading multiple secrets to memory."""
        adapter = LocalSecretAdapter()
        mapping = await adapter.aload_secrets_to_memory(
            memory=mock_memory,
            keys=["SECRET_API_KEY", "SECRET_DB_PASS"],
        )
        assert "SECRET_API_KEY" in mapping
        assert "SECRET_DB_PASS" in mapping
        assert mock_memory.aset.call_count == 2

    @pytest.mark.asyncio
    async def test_load_with_custom_prefix(self, mock_memory: AsyncMock) -> None:
        """Test loading secrets with custom memory prefix."""
        adapter = LocalSecretAdapter()
        mapping = await adapter.aload_secrets_to_memory(
            memory=mock_memory,
            prefix="mysecret:",
            keys=["SECRET_API_KEY"],
        )
        assert mapping == {"SECRET_API_KEY": "mysecret:SECRET_API_KEY"}
        mock_memory.aset.assert_called_once_with("mysecret:SECRET_API_KEY", "api-key-value")

    @pytest.mark.asyncio
    async def test_load_auto_discover_with_prefix(self, mock_memory: AsyncMock) -> None:
        """Test auto-discovering secrets with env prefix."""
        adapter = LocalSecretAdapter(env_prefix="SECRET_")
        mapping = await adapter.aload_secrets_to_memory(
            memory=mock_memory,
            keys=None,  # Auto-discover
        )
        # Should find SECRET_API_KEY and SECRET_DB_PASS
        assert "API_KEY" in mapping or "DB_PASS" in mapping

    @pytest.mark.asyncio
    async def test_load_handles_missing_secrets(self, mock_memory: AsyncMock) -> None:
        """Test that missing secrets are skipped with warning."""
        adapter = LocalSecretAdapter()
        mapping = await adapter.aload_secrets_to_memory(
            memory=mock_memory,
            keys=["SECRET_API_KEY", "NONEXISTENT_KEY"],
        )
        # Only the existing secret should be loaded
        assert "SECRET_API_KEY" in mapping
        assert "NONEXISTENT_KEY" not in mapping


class TestAlistSecretNames:
    """Tests for alist_secret_names method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up test environment variables."""
        # Clear any existing MYPREFIX_ vars and add test ones
        monkeypatch.setenv("MYPREFIX_KEY1", "value1")
        monkeypatch.setenv("MYPREFIX_KEY2", "value2")

    @pytest.mark.asyncio
    async def test_list_all_secrets(self) -> None:
        """Test listing all environment variable names."""
        adapter = LocalSecretAdapter()
        names = await adapter.alist_secret_names()
        # Should return all env vars (no prefix filter)
        assert isinstance(names, list)

    @pytest.mark.asyncio
    async def test_list_secrets_with_prefix(self) -> None:
        """Test listing secrets with prefix filter."""
        adapter = LocalSecretAdapter(env_prefix="MYPREFIX_")
        names = await adapter.alist_secret_names()
        # Should only return vars starting with MYPREFIX_, with prefix removed
        assert "KEY1" in names
        assert "KEY2" in names


class TestAhealthCheck:
    """Tests for ahealth_check method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up test environment variables."""
        monkeypatch.setenv("HEALTH_TEST_VAR", "value")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self) -> None:
        """Test health check returns healthy status."""
        adapter = LocalSecretAdapter()
        status = await adapter.ahealth_check()
        assert status.status == "healthy"
        assert status.adapter_name == "local_env"
        assert status.port_name == "secret"
        assert "env_vars_count" in status.details

    @pytest.mark.asyncio
    async def test_health_check_with_prefix(self) -> None:
        """Test health check includes prefix info."""
        adapter = LocalSecretAdapter(env_prefix="HEALTH_TEST_")
        status = await adapter.ahealth_check()
        assert status.status == "healthy"
        assert status.details["env_prefix"] == "HEALTH_TEST_"

    @pytest.mark.asyncio
    async def test_health_check_no_prefix_shows_none(self) -> None:
        """Test health check shows (none) for empty prefix."""
        adapter = LocalSecretAdapter()
        status = await adapter.ahealth_check()
        assert status.details["env_prefix"] == "(none)"


class TestSecretHiding:
    """Tests for Secret value hiding."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up test environment variables."""
        monkeypatch.setenv("SENSITIVE_KEY", "sensitive-data")

    @pytest.mark.asyncio
    async def test_secret_str_is_hidden(self) -> None:
        """Test that Secret str() is hidden."""
        adapter = LocalSecretAdapter()
        secret = await adapter.aget_secret("SENSITIVE_KEY")
        assert str(secret) == "<SECRET>"
        assert "sensitive-data" not in str(secret)

    @pytest.mark.asyncio
    async def test_secret_repr_is_hidden(self) -> None:
        """Test that Secret repr() is hidden."""
        adapter = LocalSecretAdapter()
        secret = await adapter.aget_secret("SENSITIVE_KEY")
        assert "sensitive-data" not in repr(secret)

    @pytest.mark.asyncio
    async def test_secret_get_returns_value(self) -> None:
        """Test that Secret.get() returns the actual value."""
        adapter = LocalSecretAdapter()
        secret = await adapter.aget_secret("SENSITIVE_KEY")
        assert secret.get() == "sensitive-data"
