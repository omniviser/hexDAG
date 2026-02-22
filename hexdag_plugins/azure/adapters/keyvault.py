"""Azure Key Vault adapter for hexDAG framework.

Provides secret resolution from Azure Key Vault for production deployments.
"""

import os
from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.ports.secret import SecretStore
from hexdag.kernel.types import Secret

if TYPE_CHECKING:
    from hexdag.kernel.ports.memory import Memory

logger = get_logger(__name__)


class AzureKeyVaultAdapter(SecretStore):
    """Azure Key Vault adapter for secret resolution.

    Supports both API key authentication and Azure Managed Identity for
    secure, credential-free access in Azure environments.

    Parameters
    ----------
    vault_url : str
        Azure Key Vault URL (e.g., "https://my-vault.vault.azure.net")
    use_managed_identity : bool, optional
        Use Azure Managed Identity instead of explicit credentials (default: True)
    tenant_id : str, optional
        Azure AD tenant ID (only needed if use_managed_identity=False)
    client_id : str, optional
        Azure AD client ID (only needed if use_managed_identity=False)
    client_secret : str, optional
        Azure AD client secret (only needed if use_managed_identity=False)
    cache_secrets : bool, optional
        Cache retrieved secrets in memory (default: True)
    cache_ttl : int, optional
        Cache TTL in seconds (default: 300)

    Examples
    --------
    YAML configuration with managed identity::

        spec:
          ports:
            secret:
              adapter: hexdag_plugins.azure.AzureKeyVaultAdapter
              config:
                vault_url: "https://my-vault.vault.azure.net"
                use_managed_identity: true

    Python usage::

        from hexdag_plugins.azure import AzureKeyVaultAdapter

        adapter = AzureKeyVaultAdapter(
            vault_url="https://my-vault.vault.azure.net",
            use_managed_identity=True
        )

        secret = await adapter.aget_secret("OPENAI-API-KEY")
        api_key = secret.get()  # unwrap
    """

    _hexdag_icon = "KeyRound"
    _hexdag_color = "#0078d4"  # Azure blue

    def __init__(
        self,
        vault_url: str,
        use_managed_identity: bool = True,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_secrets: bool = True,
        cache_ttl: int = 300,
    ):
        """Initialize Azure Key Vault adapter.

        Args
        ----
            vault_url: Azure Key Vault URL
            use_managed_identity: Use Managed Identity for auth (default: True)
            tenant_id: Azure AD tenant ID (for service principal auth)
            client_id: Azure AD client ID (for service principal auth)
            client_secret: Azure AD client secret (for service principal auth)
            cache_secrets: Cache retrieved secrets (default: True)
            cache_ttl: Cache TTL in seconds (default: 300)
        """
        self.vault_url = vault_url
        self.use_managed_identity = use_managed_identity
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_secrets = cache_secrets
        self.cache_ttl = cache_ttl

        self._client = None
        self._cache: dict[str, tuple[str, float]] = {}

    def _get_client(self):
        """Get or create Key Vault client."""
        if self._client is None:
            try:
                from azure.identity import (
                    ClientSecretCredential,
                    DefaultAzureCredential,
                )
                from azure.keyvault.secrets import SecretClient
            except ImportError as e:
                raise ImportError(
                    "Azure SDK not installed. Install with: "
                    "pip install azure-identity azure-keyvault-secrets"
                ) from e

            if self.use_managed_identity:
                credential = DefaultAzureCredential()
            else:
                if not all([self.tenant_id, self.client_id, self.client_secret]):
                    raise ValueError(
                        "tenant_id, client_id, and client_secret are required "
                        "when use_managed_identity=False"
                    )
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )

            self._client = SecretClient(vault_url=self.vault_url, credential=credential)

        return self._client

    def _get_from_cache(self, secret_name: str) -> str | None:
        """Get secret from cache if valid."""
        if not self.cache_secrets:
            return None

        import time

        if secret_name in self._cache:
            value, timestamp = self._cache[secret_name]
            if time.time() - timestamp < self.cache_ttl:
                return value
            del self._cache[secret_name]

        return None

    def _set_cache(self, secret_name: str, value: str) -> None:
        """Set secret in cache."""
        if self.cache_secrets:
            import time

            self._cache[secret_name] = (value, time.time())

    def _fetch_raw(self, secret_name: str) -> str:
        """Fetch raw secret value from Key Vault (sync, for internal use).

        Args
        ----
            secret_name: Name of the secret in Key Vault

        Returns
        -------
        str
            Raw secret value

        Raises
        ------
        KeyError
            If the secret does not exist
        ValueError
            If the secret value is empty/null
        RuntimeError
            If Key Vault access fails
        """
        cached = self._get_from_cache(secret_name)
        if cached is not None:
            return cached

        try:
            client = self._get_client()
            secret = client.get_secret(secret_name)

            if secret.value is None:
                raise ValueError(f"Secret '{secret_name}' has no value")

            self._set_cache(secret_name, secret.value)
            return secret.value

        except ValueError:
            raise
        except Exception as e:
            if "SecretNotFound" in str(e):
                raise KeyError(f"Secret '{secret_name}' not found in Key Vault") from e
            raise RuntimeError(f"Failed to retrieve secret '{secret_name}': {e}") from e

    # ========================================================================
    # SecretStore protocol
    # ========================================================================

    async def aget_secret(self, key: str) -> Secret:
        """Retrieve a single secret by key, wrapped in Secret.

        Args
        ----
            key: Secret identifier in Key Vault (e.g., "OPENAI-API-KEY")

        Returns
        -------
        Secret
            Secret wrapper containing the secret value

        Raises
        ------
        KeyError
            If the secret does not exist
        ValueError
            If the secret value is empty or invalid
        """
        return Secret(self._fetch_raw(key))

    async def aload_secrets_to_memory(
        self,
        memory: "Memory",
        prefix: str = "secret:",
        keys: list[str] | None = None,
    ) -> dict[str, str]:
        """Bulk load secrets from Key Vault into Memory port.

        Args
        ----
            memory: Memory port instance to store secrets in
            prefix: Key prefix for stored secrets (default: "secret:")
            keys: List of secret keys to load. If None, loads all available secrets.

        Returns
        -------
        dict[str, str]
            Mapping of original key -> memory key
        """
        mapping: dict[str, str] = {}

        if keys is None:
            keys = await self.alist_secret_names()
            logger.debug(f"Auto-discovered {len(keys)} secrets from Key Vault")

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
        """List all available secret names from Key Vault.

        Returns
        -------
        list[str]
            List of secret identifiers in the Key Vault
        """
        try:
            client = self._get_client()
            return [secret.name for secret in client.list_properties_of_secrets()]
        except Exception as e:
            raise RuntimeError(f"Failed to list secrets: {e}") from e

    async def ahealth_check(self) -> HealthStatus:
        """Check Azure Key Vault connectivity.

        Returns
        -------
            HealthStatus with connectivity details
        """
        import time

        try:
            start_time = time.time()
            client = self._get_client()

            # Try to list secrets (limited to 1) to verify connectivity
            list(client.list_properties_of_secrets())

            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                status="healthy",
                adapter_name="AzureKeyVault",
                latency_ms=latency_ms,
                details={
                    "vault_url": self.vault_url,
                    "auth_method": "managed_identity"
                    if self.use_managed_identity
                    else "service_principal",
                    "cache_enabled": self.cache_secrets,
                },
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="AzureKeyVault",
                latency_ms=0.0,
                details={"error": str(e), "vault_url": self.vault_url},
            )

    # ========================================================================
    # Additional Key Vault operations (not part of SecretStore)
    # ========================================================================

    async def aset(self, secret_name: str, value: str) -> None:
        """Set a secret in Azure Key Vault.

        Args
        ----
            secret_name: Name of the secret
            value: Secret value to store
        """
        try:
            client = self._get_client()
            client.set_secret(secret_name, value)
            self._set_cache(secret_name, value)
        except Exception as e:
            raise RuntimeError(f"Failed to set secret '{secret_name}': {e}") from e

    async def adelete(self, secret_name: str) -> None:
        """Delete a secret from Azure Key Vault.

        Args
        ----
            secret_name: Name of the secret to delete
        """
        try:
            client = self._get_client()
            client.begin_delete_secret(secret_name)
            if secret_name in self._cache:
                del self._cache[secret_name]
        except Exception as e:
            raise RuntimeError(f"Failed to delete secret '{secret_name}': {e}") from e

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()

    # ========================================================================
    # Environment variable initialization
    # ========================================================================

    async def load_to_environ(
        self,
        keys: list[str] | None = None,
        prefix: str = "",
        overwrite: bool = False,
    ) -> dict[str, str]:
        """Load secrets from Key Vault into ``os.environ``.

        This makes vault secrets available to all ``${VAR}`` resolution in
        YAML pipelines and adapter ``__init__`` methods (e.g., ``OpenAIAdapter``
        reads ``os.getenv("OPENAI_API_KEY")``).

        Key Vault uses hyphens (``OPENAI-API-KEY``) while env vars use
        underscores (``OPENAI_API_KEY``). This method normalizes automatically.

        Args
        ----
            keys: Specific secret keys to load. If None, loads all.
            prefix: Env var name prefix (e.g., "MYAPP_").
            overwrite: Overwrite existing env vars (default: False).

        Returns
        -------
        dict[str, str]
            Mapping of env var name -> status ("loaded" or "skipped")

        Examples
        --------
        Example usage::

            vault = AzureKeyVaultAdapter(vault_url="https://my-vault.vault.azure.net")
            await vault.load_to_environ(keys=["OPENAI-API-KEY", "DB-PASSWORD"])
            # os.environ["OPENAI_API_KEY"] is now set
            # os.environ["DB_PASSWORD"] is now set
        """
        results: dict[str, str] = {}

        if keys is None:
            keys = await self.alist_secret_names()
            logger.debug(f"Auto-discovered {len(keys)} secrets for env loading")

        for key in keys:
            env_var_name = f"{prefix}{key.replace('-', '_').upper()}"

            if not overwrite and env_var_name in os.environ:
                logger.debug(f"Skipping '{env_var_name}' (already set in environ)")
                results[env_var_name] = "skipped"
                continue

            try:
                value = self._fetch_raw(key)
                os.environ[env_var_name] = value
                results[env_var_name] = "loaded"
                logger.debug(f"Loaded '{key}' → env:{env_var_name}")
            except (KeyError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to load secret '{key}' to environ: {e}")
                results[env_var_name] = f"error: {e}"

        loaded = sum(1 for v in results.values() if v == "loaded")
        logger.info(f"Loaded {loaded}/{len(keys)} secrets into os.environ")
        return results

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter configuration (excluding secrets)."""
        return {
            "vault_url": self.vault_url,
            "use_managed_identity": self.use_managed_identity,
            "cache_secrets": self.cache_secrets,
            "cache_ttl": self.cache_ttl,
        }
