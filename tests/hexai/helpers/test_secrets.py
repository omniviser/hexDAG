import pytest

from hexai.helpers.secrets import Secret, get_secret


class TestSecret:
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


class TestGetSecret:
    def test_get_existing_secret(self, monkeypatch):
        """Test retrieving an existing environment variable"""
        monkeypatch.setenv("TEST_KEY", "test-value")
        secret = get_secret("TEST_KEY")
        assert secret.get() == "test-value"

    def test_missing_secret(self):
        """Test behavior when secret is not found"""
        with pytest.raises(ValueError) as exc_info:
            get_secret("NONEXISTENT_KEY")
        assert "not found in environment variables" in str(exc_info.value)

    def test_direct_access_prevention(self):
        """Test that direct access to secret value is prevented"""
        secret = Secret("protected-value")
        with pytest.raises(AttributeError):
            # Try to access mangled name
            secret.__value  # type: ignore
        with pytest.raises(AttributeError):
            # Try to access with single underscore
            secret._value  # type: ignore
