"""Simple secret resolution for adapters.

This module provides a clean way to declare secrets in adapter __init__ signatures
without complex Config classes. Secrets are resolved from environment variables
or Memory port automatically.

Example:
    @adapter("llm", name="openai")
    class OpenAIAdapter:
        def __init__(
            self,
            api_key: str = secret(env="OPENAI_API_KEY"),
            model: str = "gpt-4"
        ):
            self.api_key = api_key
            self.model = model
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

from hexdag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SecretDescriptor:
    """Descriptor for a secret parameter.

    This is used as a default value in __init__ to mark a parameter
    as a secret that should be auto-resolved.

    Attributes
    ----------
    env_var : str
        Environment variable name to resolve from
    memory_key : str | None
        Key in Memory port (defaults to env_var with "secret:" prefix)
    required : bool
        Whether this secret is required
    description : str
        Human-readable description
    """

    env_var: str
    memory_key: str | None = None
    required: bool = True
    description: str = ""

    def resolve(self, memory: Any = None) -> str | None:
        """Resolve secret value from environment or memory.

        Resolution order:
        1. Environment variable
        2. Memory port (with "secret:" prefix)
        3. None (if not required)

        Parameters
        ----------
        memory : Any, optional
            Memory port instance to read from

        Returns
        -------
        str | None
            Resolved secret value or None

        Raises
        ------
        ValueError
            If secret is required but not found
        """
        # Try environment first
        if value := os.getenv(self.env_var):
            logger.debug(f"Resolved secret from env: {self.env_var}")
            return value

        # Try memory
        if memory:
            memory_key = self.memory_key or f"secret:{self.env_var}"
            try:
                if hasattr(memory, "get"):
                    value = memory.get(memory_key)
                elif hasattr(memory, "aget"):
                    # Async memory - can't resolve here
                    logger.warning(
                        f"Memory port is async, cannot resolve {memory_key} in __init__. "
                        "Consider using environment variable."
                    )
                else:
                    value = None

                if value:
                    logger.debug(f"Resolved secret from memory: {memory_key}")
                    # Handle SecretStr from Pydantic or plain string
                    if hasattr(value, "get_secret_value"):
                        return value.get_secret_value()  # type: ignore[no-any-return]
                    return str(value)
            except Exception as e:
                logger.debug(f"Failed to read from memory: {e}")

        # Not found
        if self.required:
            raise ValueError(
                f"Required secret '{self.env_var}' not found. "
                f"Set environment variable {self.env_var} or provide in memory."
            )

        return None


def secret(
    env: str, memory_key: str | None = None, required: bool = True, description: str = ""
) -> SecretDescriptor:
    """Mark a parameter as a secret that should be auto-resolved.

    Use this as a default value in __init__ to declare secrets.
    The @adapter decorator will automatically resolve these.

    Parameters
    ----------
    env : str
        Environment variable name (e.g., "OPENAI_API_KEY")
    memory_key : str | None, optional
        Alternative key in Memory port. If None, uses "secret:{env}"
    required : bool, default=True
        Whether this secret is required
    description : str, default=""
        Human-readable description for docs/CLI

    Returns
    -------
    SecretDescriptor
        Secret descriptor that will be resolved by @adapter decorator

    Examples
    --------
    >>> @adapter("llm", name="openai")  # doctest: +SKIP
    ... class OpenAIAdapter:
    ...     def __init__(
    ...         self,
    ...         api_key: str = secret(env="OPENAI_API_KEY", description="OpenAI API key"),
    ...         model: str = "gpt-4"
    ...     ):
    ...         self.api_key = api_key
    ...         self.model = model
    """
    return SecretDescriptor(
        env_var=env, memory_key=memory_key, required=required, description=description
    )


def resolve_secrets_in_kwargs(
    cls: type, kwargs: dict[str, Any], memory: Any = None
) -> dict[str, Any]:
    """Resolve secrets in kwargs based on __init__ signature.

    Scans the __init__ signature for SecretDescriptor defaults
    and resolves them from environment or memory.

    Parameters
    ----------
    cls : type
        Class to inspect
    kwargs : dict[str, Any]
        Keyword arguments passed to __init__
    memory : Any, optional
        Memory port for secret resolution

    Returns
    -------
    dict[str, Any]
        Updated kwargs with resolved secrets

    Examples
    --------
    >>> kwargs = resolve_secrets_in_kwargs(OpenAIAdapter, {}, memory)  # doctest: +SKIP
    >>> # If OPENAI_API_KEY is set: kwargs = {"api_key": "sk-..."}
    """
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Check if default value is a SecretDescriptor
        if isinstance(param.default, SecretDescriptor):
            secret_desc = param.default

            # Skip if already provided in kwargs
            if param_name in kwargs and kwargs[param_name] is not None:
                logger.debug(f"Secret '{param_name}' provided explicitly")
                continue

            # Resolve the secret
            try:
                resolved_value = secret_desc.resolve(memory=memory)
                if resolved_value is not None:
                    kwargs[param_name] = resolved_value
                    logger.debug(f"Resolved secret '{param_name}'")
            except ValueError as e:
                # Required secret not found
                logger.error(f"Failed to resolve secret '{param_name}': {e}")
                raise

    return kwargs


def extract_secrets_from_signature(cls: type) -> dict[str, SecretDescriptor]:
    """Extract all secret declarations from __init__ signature.

    This is used by CLI to show which secrets are required.

    Parameters
    ----------
    cls : type
        Class to inspect

    Returns
    -------
    dict[str, SecretDescriptor]
        Mapping of parameter name to SecretDescriptor

    Examples
    --------
    >>> secrets = extract_secrets_from_signature(OpenAIAdapter)  # doctest: +SKIP
    >>> # {"api_key": SecretDescriptor(env_var="OPENAI_API_KEY", ...)}
    """
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    secrets = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        if isinstance(param.default, SecretDescriptor):
            secrets[param_name] = param.default

    return secrets
