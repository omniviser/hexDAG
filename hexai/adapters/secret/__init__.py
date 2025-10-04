"""Secret management adapters."""

from hexai.core.types import Secret

from .local_secret_adapter import LocalSecretAdapter

__all__ = ["LocalSecretAdapter", "Secret"]
