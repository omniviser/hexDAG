"""Local environment variable based secret adapter."""

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from hexai.core.configurable import AdapterConfig, ConfigurableAdapter
from hexai.core.logging import get_logger
from hexai.core.registry import adapter
from hexai.core.types import Secret

if TYPE_CHECKING:
    from hexai.core.ports.healthcheck import HealthStatus
    from hexai.core.ports.memory import Memory

logger = get_logger(__name__)


@adapter(
    name="local_env",
    implements_port="secret",
    namespace="core",
    description="Local environment variable based secret management",
)
class LocalSecretAdapter(ConfigurableAdapter):
    """Local secret adapter that reads from environment variables.

    This adapter implements the SecretPort interface using local environment
    variables as the secret source. It's useful for:
    - Development and testing
    - CI/CD pipelines
    - Simple deployments without external secret managers

    The adapter wraps secrets in the Secret class to prevent accidental logging.

    Examples
    --------
    Basic usage::

        # Create adapter
        secrets = LocalSecretAdapter()

        # Get single secret
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

    class Config(AdapterConfig):
        """Configuration schema for Local Secret adapter."""

        env_prefix: str = Field(
            default="",
            description="Optional prefix for environment variable names (e.g., 'MYAPP_')",
        )
        allow_empty: bool = Field(
            default=False, description="Allow empty secret values (default: False)"
        )

    config: Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize local secret adapter.

        Args
        ----
            **kwargs: Configuration options (env_prefix, allow_empty)
        """
        super().__init__(**kwargs)

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
        # Add prefix if configured
        env_var_name = f"{self.config.env_prefix}{key}"

        # Get from environment
        value = os.getenv(env_var_name)

        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in environment variables (looked for: {env_var_name})"
            )

        if value == "" and not self.config.allow_empty:
            raise ValueError(
                f"Secret '{key}' cannot be empty (set allow_empty=True to allow empty secrets)"
            )

        logger.debug(f"Retrieved secret '{key}' from environment")
        return Secret(value)

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
                key.removeprefix(self.config.env_prefix)
                for key in os.environ
                if key.startswith(self.config.env_prefix)
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
        # Get all env vars matching prefix
        names = [
            key.removeprefix(self.config.env_prefix)
            for key in os.environ
            if key.startswith(self.config.env_prefix)
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
        from hexai.core.ports.healthcheck import HealthStatus

        try:
            # Count available env vars
            names = await self.alist_secret_names()
            return HealthStatus(
                status="healthy",
                adapter_name="local_env",
                port_name="secret",
                details={
                    "env_vars_count": len(names),
                    "env_prefix": self.config.env_prefix or "(none)",
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
