"""Azure Key Vault adapter for hexDAG framework.

Provides secret resolution from Azure Key Vault for production deployments.
"""

from typing import Any

from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.registry import adapter


@adapter(
    "secret",
    name="azure_keyvault",
    secrets={},  # No secrets needed - uses managed identity or explicit credentials
)
class AzureKeyVaultAdapter:
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
              adapter: azure_keyvault
              config:
                vault_url: "https://my-vault.vault.azure.net"
                use_managed_identity: true

    YAML configuration with service principal::

        spec:
          ports:
            secret:
              adapter: azure_keyvault
              config:
                vault_url: "https://my-vault.vault.azure.net"
                use_managed_identity: false
                tenant_id: ${AZURE_TENANT_ID}
                client_id: ${AZURE_CLIENT_ID}
                client_secret: ${AZURE_CLIENT_SECRET}

    Python usage::

        from hexdag_plugins.azure import AzureKeyVaultAdapter

        # With managed identity (recommended for Azure deployments)
        adapter = AzureKeyVaultAdapter(
            vault_url="https://my-vault.vault.azure.net",
            use_managed_identity=True
        )

        # Retrieve secrets
        api_key = await adapter.aget("OPENAI-API-KEY")
        db_password = await adapter.aget("DB-PASSWORD")

        # Batch retrieval
        secrets = await adapter.aget_batch(["SECRET1", "SECRET2"])
    """

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

    async def aget(self, secret_name: str) -> str:
        """Retrieve a secret from Azure Key Vault.

        Args
        ----
            secret_name: Name of the secret in Key Vault

        Returns
        -------
            Secret value as string

        Raises
        ------
            ValueError: If secret not found
            RuntimeError: If Key Vault access fails
        """
        # Check cache first
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

        except Exception as e:
            if "SecretNotFound" in str(e):
                raise ValueError(f"Secret '{secret_name}' not found in Key Vault") from e
            raise RuntimeError(f"Failed to retrieve secret '{secret_name}': {e}") from e

    async def aget_batch(self, secret_names: list[str]) -> dict[str, str]:
        """Retrieve multiple secrets from Azure Key Vault.

        Args
        ----
            secret_names: List of secret names to retrieve

        Returns
        -------
            Dictionary mapping secret names to values
        """
        results = {}
        for name in secret_names:
            try:
                results[name] = await self.aget(name)
            except ValueError:  # noqa: SIM105 - contextlib.suppress doesn't work with async
                # Skip secrets that don't exist
                pass
        return results

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

    async def alist(self) -> list[str]:
        """List all secret names in the Key Vault.

        Returns
        -------
            List of secret names
        """
        try:
            client = self._get_client()
            return [secret.name for secret in client.list_properties_of_secrets()]
        except Exception as e:
            raise RuntimeError(f"Failed to list secrets: {e}") from e

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter configuration (excluding secrets)."""
        return {
            "vault_url": self.vault_url,
            "use_managed_identity": self.use_managed_identity,
            "cache_secrets": self.cache_secrets,
            "cache_ttl": self.cache_ttl,
        }
