"""Port interface for secret management (KeyVault, AWS Secrets Manager, etc.)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hexai.core.registry.decorators import port

if TYPE_CHECKING:
    from hexai.core.ports.healthcheck import HealthStatus
    from hexai.core.ports.memory import Memory
    from hexai.helpers.secrets import Secret


@port(
    name="secret",
    namespace="core",
)
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
        >>> # Fetch secret from KeyVault
        >>> secret = await keyvault.aget_secret("OPENAI_API_KEY")
        >>> api_key = secret.get()  # Unwrap the secret value
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
        >>> # Load specific secrets
        >>> mapping = await keyvault.aload_secrets_to_memory(
        ...     memory=memory,
        ...     keys=["OPENAI_API_KEY", "DATABASE_PASSWORD"]
        ... )
        >>> # Returns: {"OPENAI_API_KEY": "secret:OPENAI_API_KEY", ...}

        >>> # Load all secrets
        >>> mapping = await keyvault.aload_secrets_to_memory(memory=memory)

        >>> # Retrieve in nodes
        >>> api_key = await memory.aget("secret:OPENAI_API_KEY")
        """
        ...

    async def alist_secret_names(self) -> list[str]:
        """List all available secret names (optional).

        Returns
        -------
        list[str]
            List of secret identifiers available in this secret store

        Examples
        --------
        >>> names = await keyvault.alist_secret_names()
        >>> # ["OPENAI_API_KEY", "DATABASE_PASSWORD", "STRIPE_KEY"]
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
        >>> # Azure KeyVault health check
        >>> status = await keyvault.ahealth_check()
        >>> status.status  # "healthy", "degraded", or "unhealthy"
        >>> status.details  # {"vault_url": "...", "authenticated": True}
        """
        ...
