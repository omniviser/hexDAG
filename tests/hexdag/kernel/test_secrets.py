"""Tests for the secrets module.

This module tests secret resolution for adapters.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from hexdag.kernel.secrets import (
    SecretDescriptor,
    extract_secrets_from_signature,
    resolve_secrets_in_kwargs,
    secret,
)


class TestSecretDescriptor:
    """Tests for SecretDescriptor."""

    def test_basic_creation(self) -> None:
        """Test creating a basic secret descriptor."""
        desc = SecretDescriptor(env_var="API_KEY")
        assert desc.env_var == "API_KEY"
        assert desc.memory_key is None
        assert desc.required is True
        assert desc.description == ""

    def test_full_creation(self) -> None:
        """Test creating a fully specified secret descriptor."""
        desc = SecretDescriptor(
            env_var="API_KEY",
            memory_key="secret:api_key",
            required=False,
            description="API key for service",
        )
        assert desc.env_var == "API_KEY"
        assert desc.memory_key == "secret:api_key"
        assert desc.required is False
        assert desc.description == "API key for service"

    def test_resolve_from_env(self) -> None:
        """Test resolving secret from environment variable."""
        os.environ["TEST_SECRET"] = "secret_value"
        try:
            desc = SecretDescriptor(env_var="TEST_SECRET")
            value = desc.resolve()
            assert value == "secret_value"
        finally:
            del os.environ["TEST_SECRET"]

    def test_resolve_from_memory(self) -> None:
        """Test resolving secret from memory port."""
        # Ensure env var not set
        os.environ.pop("MEMORY_SECRET", None)

        desc = SecretDescriptor(env_var="MEMORY_SECRET")

        # Create mock memory with get method
        memory = MagicMock()
        memory.get.return_value = "memory_value"

        value = desc.resolve(memory=memory)
        assert value == "memory_value"
        memory.get.assert_called_once_with("secret:MEMORY_SECRET")

    def test_resolve_from_memory_custom_key(self) -> None:
        """Test resolving secret from memory with custom key."""
        os.environ.pop("CUSTOM_SECRET", None)

        desc = SecretDescriptor(env_var="CUSTOM_SECRET", memory_key="custom_key")

        memory = MagicMock()
        memory.get.return_value = "custom_value"

        value = desc.resolve(memory=memory)
        assert value == "custom_value"
        memory.get.assert_called_once_with("custom_key")

    def test_resolve_from_memory_with_secret_value(self) -> None:
        """Test resolving secret that has get_secret_value method."""
        os.environ.pop("PYDANTIC_SECRET", None)

        desc = SecretDescriptor(env_var="PYDANTIC_SECRET")

        # Mock SecretStr-like object
        secret_obj = MagicMock()
        secret_obj.get_secret_value.return_value = "actual_secret"

        memory = MagicMock()
        memory.get.return_value = secret_obj

        value = desc.resolve(memory=memory)
        assert value == "actual_secret"

    def test_resolve_required_not_found(self) -> None:
        """Test that missing required secret raises error."""
        os.environ.pop("MISSING_SECRET", None)

        desc = SecretDescriptor(env_var="MISSING_SECRET", required=True)

        with pytest.raises(ValueError, match="Required secret.*not found"):
            desc.resolve()

    def test_resolve_optional_not_found(self) -> None:
        """Test that missing optional secret returns None."""
        os.environ.pop("OPTIONAL_SECRET", None)

        desc = SecretDescriptor(env_var="OPTIONAL_SECRET", required=False)

        value = desc.resolve()
        assert value is None

    def test_resolve_env_takes_precedence(self) -> None:
        """Test that environment variable takes precedence over memory."""
        os.environ["PRECEDENCE_SECRET"] = "env_value"
        try:
            desc = SecretDescriptor(env_var="PRECEDENCE_SECRET")

            memory = MagicMock()
            memory.get.return_value = "memory_value"

            value = desc.resolve(memory=memory)
            assert value == "env_value"
            # Memory should not be called if env found
            memory.get.assert_not_called()
        finally:
            del os.environ["PRECEDENCE_SECRET"]

    def test_resolve_memory_failure_continues(self) -> None:
        """Test that memory failure doesn't raise for optional secrets."""
        os.environ.pop("FAILING_SECRET", None)

        desc = SecretDescriptor(env_var="FAILING_SECRET", required=False)

        memory = MagicMock()
        memory.get.side_effect = Exception("Memory error")

        value = desc.resolve(memory=memory)
        assert value is None

    def test_resolve_async_memory_warning(self) -> None:
        """Test that async memory produces warning."""
        os.environ.pop("ASYNC_SECRET", None)

        desc = SecretDescriptor(env_var="ASYNC_SECRET", required=False)

        # Mock async memory (has aget but not get)
        memory = MagicMock(spec=["aget"])

        value = desc.resolve(memory=memory)
        assert value is None


class TestSecretFunction:
    """Tests for the secret() helper function."""

    def test_basic_secret(self) -> None:
        """Test creating basic secret descriptor."""
        desc = secret(env="API_KEY")
        assert isinstance(desc, SecretDescriptor)
        assert desc.env_var == "API_KEY"
        assert desc.required is True

    def test_optional_secret(self) -> None:
        """Test creating optional secret."""
        desc = secret(env="OPTIONAL", required=False)
        assert desc.required is False

    def test_secret_with_memory_key(self) -> None:
        """Test creating secret with custom memory key."""
        desc = secret(env="KEY", memory_key="custom:key")
        assert desc.memory_key == "custom:key"

    def test_secret_with_description(self) -> None:
        """Test creating secret with description."""
        desc = secret(env="KEY", description="My secret key")
        assert desc.description == "My secret key"


class TestResolveSecretsInKwargs:
    """Tests for resolve_secrets_in_kwargs function."""

    def test_class_without_secrets(self) -> None:
        """Test class with no secret parameters."""

        class NoSecrets:
            def __init__(self, name: str, value: int = 10):
                pass

        kwargs = resolve_secrets_in_kwargs(NoSecrets, {"name": "test"})
        assert kwargs == {"name": "test"}

    def test_class_with_secret_from_env(self) -> None:
        """Test resolving secret from environment."""
        os.environ["CLASS_SECRET"] = "resolved"
        try:

            class WithSecret:
                def __init__(self, api_key: str = secret(env="CLASS_SECRET"), model: str = "gpt-4"):
                    pass

            kwargs = resolve_secrets_in_kwargs(WithSecret, {"model": "claude"})
            assert kwargs == {"api_key": "resolved", "model": "claude"}
        finally:
            del os.environ["CLASS_SECRET"]

    def test_explicit_value_not_overwritten(self) -> None:
        """Test that explicitly provided values are not overwritten."""
        os.environ["EXPLICIT_SECRET"] = "from_env"
        try:

            class WithSecret:
                def __init__(self, api_key: str = secret(env="EXPLICIT_SECRET")):
                    pass

            kwargs = resolve_secrets_in_kwargs(WithSecret, {"api_key": "explicit_value"})
            assert kwargs == {"api_key": "explicit_value"}
        finally:
            del os.environ["EXPLICIT_SECRET"]

    def test_required_secret_missing(self) -> None:
        """Test that missing required secret raises error."""
        os.environ.pop("MISSING_CLASS_SECRET", None)

        class RequiredSecret:
            def __init__(self, api_key: str = secret(env="MISSING_CLASS_SECRET")):
                pass

        with pytest.raises(ValueError, match="Required secret"):
            resolve_secrets_in_kwargs(RequiredSecret, {})

    def test_optional_secret_missing(self) -> None:
        """Test that missing optional secret doesn't add to kwargs."""
        os.environ.pop("OPTIONAL_CLASS_SECRET", None)

        class OptionalSecret:
            def __init__(self, api_key: str = secret(env="OPTIONAL_CLASS_SECRET", required=False)):
                pass

        kwargs = resolve_secrets_in_kwargs(OptionalSecret, {})
        assert "api_key" not in kwargs


class TestExtractSecretsFromSignature:
    """Tests for extract_secrets_from_signature function."""

    def test_class_without_secrets(self) -> None:
        """Test extracting from class with no secrets."""

        class NoSecrets:
            def __init__(self, name: str):
                pass

        secrets = extract_secrets_from_signature(NoSecrets)
        assert secrets == {}

    def test_class_with_one_secret(self) -> None:
        """Test extracting single secret."""

        class OneSecret:
            def __init__(self, api_key: str = secret(env="API_KEY")):
                pass

        secrets = extract_secrets_from_signature(OneSecret)
        assert "api_key" in secrets
        assert secrets["api_key"].env_var == "API_KEY"

    def test_class_with_multiple_secrets(self) -> None:
        """Test extracting multiple secrets."""

        class MultiSecrets:
            def __init__(
                self,
                api_key: str = secret(env="API_KEY"),
                secret_token: str = secret(env="TOKEN"),
                name: str = "default",  # Not a secret
            ):
                pass

        secrets = extract_secrets_from_signature(MultiSecrets)
        assert len(secrets) == 2
        assert "api_key" in secrets
        assert "secret_token" in secrets
        assert "name" not in secrets

    def test_class_with_mixed_params(self) -> None:
        """Test class with mix of secrets and regular params."""

        class MixedParams:
            def __init__(
                self,
                name: str,
                api_key: str = secret(env="KEY"),
                count: int = 10,
                password: str = secret(env="PASSWORD", required=False),
            ):
                pass

        secrets = extract_secrets_from_signature(MixedParams)
        assert len(secrets) == 2
        assert secrets["api_key"].required is True
        assert secrets["password"].required is False
