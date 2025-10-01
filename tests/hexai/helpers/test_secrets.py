import pytest

from hexai.helpers.secrets import Secret


class TestSecret:
    """Test suite for Secret class functionality."""

    def test_secret_creation(self):
        """Test basic secret creation and retrieval"""
        secret = Secret("test-value")
        assert secret.get() == "test-value"
        assert str(secret) == "<SECRET>"
        assert repr(secret) == "<SECRET>"

    def test_secret_in_string_formatting(self):
        """Test that secrets are protected in string formatting"""
        secret = Secret("sensitive-data")
        formatted = f"My secret is: {secret}"
        assert "sensitive-data" not in formatted
        assert "<SECRET>" in formatted

    def test_secret_in_dict(self):
        """Test that secrets are protected in dictionary representations"""
        secret = Secret("api-key")
        headers = {"Authorization": f"Bearer {secret}"}
        assert "api-key" not in str(headers)
        assert "<SECRET>" in str(headers)

    @pytest.fixture
    def env_setup(self, monkeypatch):
        """Setup test environment variables."""
        monkeypatch.setenv("PRESENT_SECRET", "actual-value")
        monkeypatch.setenv("EMPTY_SECRET", "")
        # MISSING_SECRET is intentionally not set

    def test_retrieve_present_secret(self, env_setup):
        """Test retrieving an existing non-empty secret."""
        secret = Secret.retrieve_secret_from_env("PRESENT_SECRET")
        assert secret.get() == "actual-value"

    def test_retrieve_empty_secret(self, env_setup):
        """Test handling of empty string secrets."""
        with pytest.raises(ValueError) as exc_info:
            Secret.retrieve_secret_from_env("EMPTY_SECRET")
        assert "Secret value cannot be empty" in str(exc_info.value)

    def test_retrieve_missing_secret(self, env_setup):
        """Test retrieving a missing secret."""
        with pytest.raises(KeyError) as exc_info:
            Secret.retrieve_secret_from_env("MISSING_SECRET")
        assert "not found in environment variables" in str(exc_info.value)
