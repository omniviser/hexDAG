"""Port interface for secret management (KeyVault, AWS Secrets Manager, etc.)."""

import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.kernel.ports.healthcheck import HealthStatus
    from hexdag.kernel.ports.memory import Memory
    from hexdag.kernel.types import Secret

logger = get_logger(__name__)


@runtime_checkable
class SecretPort(Protocol):
    """Port interface for secret/credential management systems.

    This port abstracts access to secret management services like:
    - Azure KeyVault
    - AWS Secrets Manager
    - HashiCorp Vault
    - Google Secret Manager
    - Environment variables

    Secrets are returned as Secret[str] objects to prevent accidental logging.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - load_to_environ(): Load secrets into os.environ (has default impl)
    - ahealth_check(): Verify secret service connectivity and authentication
    """

    @abstractmethod
    async def aget_secret(self, key: str) -> "Secret":
        """Retrieve a single secret by key.

        Args
        ----
            key: Secret identifier (e.g., "OPENAI_API_KEY", "database/password")

        Returns
        -------
        Secret
            Secret wrapper containing the secret value. Use .get() to retrieve the value.

        Raises
        ------
        KeyError
            If the secret does not exist
        ValueError
            If the secret value is empty or invalid

        Examples
        --------
        Example usage::

            # Fetch secret from KeyVault
            secret = await keyvault.aget_secret("OPENAI_API_KEY")
            api_key = secret.get()  # Unwrap the secret value
        """
        ...

    @abstractmethod
    async def aload_secrets_to_memory(
        self,
        memory: "Memory",
        prefix: str = "secret:",
        keys: list[str] | None = None,
    ) -> dict[str, str]:
        """Bulk load secrets into Memory port with a prefix.

        This is the primary method for pre-DAG hooks to inject secrets
        into the pipeline's memory for nodes to access.

        Args
        ----
            memory: Memory port instance to store secrets in
            prefix: Key prefix for stored secrets (default: "secret:")
            keys: List of secret keys to load. If None, loads all available secrets.

        Returns
        -------
        dict[str, str]
            Mapping of original key → memory key (e.g., {"API_KEY": "secret:API_KEY"})

        Examples
        --------
        Example usage::

            # Load specific secrets
            mapping = await keyvault.aload_secrets_to_memory(
            memory=memory,
            keys=["OPENAI_API_KEY", "DATABASE_PASSWORD"]
            )
            # Returns: {"OPENAI_API_KEY": "secret:OPENAI_API_KEY", ...}

            # Load all secrets
            mapping = await keyvault.aload_secrets_to_memory(memory=memory)

            # Retrieve in nodes
            api_key = await memory.aget("secret:OPENAI_API_KEY")
        """
        ...

    async def load_to_environ(
        self,
        keys: list[str] | None = None,
        prefix: str = "",
        overwrite: bool = False,
    ) -> dict[str, str]:
        """Load secrets into ``os.environ`` for ``${VAR}`` resolution.

        This enables YAML pipelines to use ``${OPENAI_API_KEY}`` in port
        configs — the secret is fetched from the vault and placed in
        ``os.environ`` *before* adapters are instantiated.

        Key names are normalised: hyphens become underscores and the result
        is upper-cased (e.g. ``OPENAI-API-KEY`` → ``OPENAI_API_KEY``).

        Adapters may override this for bulk-loading optimisations.

        Args
        ----
            keys: Secret keys to load. If None, loads all (via ``alist_secret_names``).
            prefix: Env-var name prefix (e.g. ``"MYAPP_"``).
            overwrite: Overwrite existing env vars (default: False).

        Returns
        -------
        dict[str, str]
            Mapping of env-var name → status (``"loaded"`` or ``"skipped"``).

        Examples
        --------
        Example usage::

            vault = AzureKeyVaultAdapter(vault_url="https://my-vault.vault.azure.net")
            await vault.load_to_environ(keys=["OPENAI-API-KEY"])
            # os.environ["OPENAI_API_KEY"] is now set
        """
        results: dict[str, str] = {}

        if keys is None:
            keys = await self.alist_secret_names()

        for key in keys:
            env_var_name = f"{prefix}{key.replace('-', '_').upper()}"

            if not overwrite and env_var_name in os.environ:
                logger.debug(f"Skipping '{env_var_name}' (already set)")
                results[env_var_name] = "skipped"
                continue

            try:
                secret = await self.aget_secret(key)
                os.environ[env_var_name] = secret.get()
                results[env_var_name] = "loaded"
                logger.debug(f"Loaded '{key}' → env:{env_var_name}")
            except (KeyError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to load secret '{key}' to environ: {e}")
                results[env_var_name] = f"error: {e}"

        loaded = sum(1 for v in results.values() if v == "loaded")
        logger.info(f"Loaded {loaded}/{len(keys)} secrets into os.environ")
        return results

    async def alist_secret_names(self) -> list[str]:
        """List all available secret names (optional).

        Returns
        -------
        list[str]
            List of secret identifiers available in this secret store

        Examples
        --------
        Example usage::

            names = await keyvault.alist_secret_names()
            # ["OPENAI_API_KEY", "DATABASE_PASSWORD", "STRIPE_KEY"]
        """
        ...

    async def ahealth_check(self) -> "HealthStatus":
        """Check secret service health and connectivity (optional).

        Adapters should verify:
        - Authentication/authorization status
        - Service connectivity
        - Access permissions

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about secret service connectivity

        Examples
        --------
        Example usage::

            # Azure KeyVault health check
            status = await keyvault.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.details  # {"vault_url": "...", "authenticated": True}
        """
        ...
