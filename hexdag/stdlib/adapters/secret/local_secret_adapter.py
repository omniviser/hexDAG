"""Local environment variable based secret adapter."""

import os
from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.secret import SecretStore
from hexdag.kernel.types import Secret

if TYPE_CHECKING:
    from hexdag.kernel.ports.healthcheck import HealthStatus
    from hexdag.kernel.ports.memory import Memory

logger = get_logger(__name__)


class LocalSecretAdapter(SecretStore):
    """Local secret adapter that reads from environment variables.

    This adapter implements the SecretStore interface using local environment
    variables as the secret source. It's useful for:
    - Development and testing
    - CI/CD pipelines
    - Simple deployments without external secret managers

    The adapter wraps secrets in the Secret class to prevent accidental logging.

    Examples
    --------
    Basic usage::

        secrets = LocalSecretAdapter()

        api_key = await secrets.aget_secret("OPENAI_API_KEY")
        print(api_key)  # <SECRET> (hidden)
        print(api_key.get())  # "sk-..." (actual value)

        # Load secrets into Memory for orchestrator
        mapping = await secrets.aload_secrets_to_memory(
            memory=memory,
            keys=["OPENAI_API_KEY", "DATABASE_PASSWORD"]
        )
        # Returns: {"OPENAI_API_KEY": "secret:OPENAI_API_KEY", ...}

    With prefix filtering::

        # Only load secrets with specific prefix
        secrets = LocalSecretAdapter(env_prefix="MYAPP_")
        # Will look for MYAPP_OPENAI_API_KEY, etc.
    """

    # Type annotations for attributes
    env_prefix: str
    allow_empty: bool
    _cache: dict[str, Secret]

    def __init__(
        self,
        env_prefix: str = "",
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize local secret adapter.

        Args
        ----
            env_prefix: Prefix for environment variable names (e.g., "MYAPP_").
            allow_empty: Allow empty secret values. Default: False.
            **kwargs: Additional options for forward compatibility.
        """
        self.env_prefix = env_prefix
        self.allow_empty = allow_empty

        self._cache: dict[str, Secret] = {}

    async def aget_secret(self, key: str) -> Secret:
        """Retrieve a single secret from environment variables.

        Args
        ----
            key: Secret identifier (environment variable name)

        Returns
        -------
        Secret
            Secret wrapper containing the secret value

        Raises
        ------
        KeyError
            If the secret is not found in environment variables
        ValueError
            If the secret value is empty (unless allow_empty=True)

        Examples
        --------
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> adapter = LocalSecretAdapter()
        >>> secret = await adapter.aget_secret("OPENAI_API_KEY")  # doctest: +SKIP
        >>> print(secret)  # doctest: +SKIP
        <SECRET>
        >>> secret.get()  # doctest: +SKIP
        'sk-test'
        """
        # Check cache first (environment variables don't change at runtime)
        if key in self._cache:
            logger.debug(f"Retrieved secret '{key}' from cache")
            return self._cache[key]

        env_var_name = f"{self.env_prefix}{key}"

        value = os.getenv(env_var_name)

        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in environment variables (looked for: {env_var_name})"
            )

        if value == "" and not self.allow_empty:
            raise ValueError(
                f"Secret '{key}' cannot be empty (set allow_empty=True to allow empty secrets)"
            )

        logger.debug(f"Retrieved secret '{key}' from environment")
        secret = Secret(value)

        self._cache[key] = secret

        return secret

    async def aload_secrets_to_memory(
        self,
        memory: "Memory",
        prefix: str = "secret:",
        keys: list[str] | None = None,
    ) -> dict[str, str]:
        """Bulk load secrets from environment into Memory port.

        Args
        ----
            memory: Memory port instance to store secrets in
            prefix: Key prefix for stored secrets (default: "secret:")
            keys: List of secret keys to load. If None, loads all env vars.

        Returns
        -------
        dict[str, str]
            Mapping of original key → memory key

        Examples
        --------
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> os.environ["DATABASE_PASSWORD"] = "pass123"
        >>> adapter = LocalSecretAdapter()
        >>> # Load specific secrets
        >>> mapping = await adapter.aload_secrets_to_memory(  # doctest: +SKIP
        ...     memory=memory,
        ...     keys=["OPENAI_API_KEY"]
        ... )
        >>> mapping  # doctest: +SKIP
        {'OPENAI_API_KEY': 'secret:OPENAI_API_KEY'}
        """
        mapping: dict[str, str] = {}

        if keys is None:
            # Load all environment variables (with prefix if configured)
            keys = [
                key.removeprefix(self.env_prefix)
                for key in os.environ
                if key.startswith(self.env_prefix)
            ]
            logger.debug(f"Auto-discovered {len(keys)} environment variables")

        # Load each secret
        loaded_count = 0
        for key in keys:
            try:
                secret = await self.aget_secret(key)
                memory_key = f"{prefix}{key}"
                await memory.aset(memory_key, secret.get())
                mapping[key] = memory_key
                loaded_count += 1
                logger.debug(f"Loaded secret '{key}' → '{memory_key}'")
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to load secret '{key}': {e}")
                continue

        logger.info(f"Loaded {loaded_count}/{len(keys)} secrets into Memory with prefix '{prefix}'")
        return mapping

    async def alist_secret_names(self) -> list[str]:
        """List all available secret names from environment variables.

        Returns
        -------
        list[str]
            List of environment variable names (with prefix removed)

        Examples
        --------
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> os.environ["DATABASE_PASSWORD"] = "pass123"
        >>> adapter = LocalSecretAdapter()
        >>> await adapter.alist_secret_names()  # doctest: +SKIP
        ['OPENAI_API_KEY', 'DATABASE_PASSWORD', ...]
        """
        names = [
            key.removeprefix(self.env_prefix)
            for key in os.environ
            if key.startswith(self.env_prefix)
        ]
        logger.debug(f"Found {len(names)} environment variables")
        return names

    async def ahealth_check(self) -> "HealthStatus":
        """Check health status of local environment variable access.

        Returns
        -------
        HealthStatus
            Health status with environment variable count

        Examples
        --------
        >>> adapter = LocalSecretAdapter()
        >>> status = await adapter.ahealth_check()  # doctest: +SKIP
        >>> status.status  # doctest: +SKIP
        'healthy'
        """
        from hexdag.kernel.ports.healthcheck import HealthStatus

        try:
            # Count available env vars
            names = await self.alist_secret_names()
            return HealthStatus(
                status="healthy",
                adapter_name="local_env",
                port_name="secret",
                details={
                    "env_vars_count": len(names),
                    "env_prefix": self.env_prefix or "(none)",
                },
            )
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="local_env",
                port_name="secret",
                error=e,
                details={
                    "error_message": str(e),
                },
            )
