"""Secret management adapters."""

from hexdag.core.types import Secret

from .local_secret_adapter import LocalSecretAdapter

__all__ = ["LocalSecretAdapter", "Secret"]
