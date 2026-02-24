"""Secret manager for loading and cleaning up secrets in Memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.kernel.ports.memory import Memory
    from hexdag.kernel.ports.secret import SecretStore

logger = get_logger(__name__)


class SecretManager:
    """Manages secret injection and cleanup lifecycle.

    Responsibilities:
    - Load secrets from SecretStore into Memory
    - Track loaded secret keys per pipeline
    - Clean up secrets after pipeline execution

    Examples
    --------
    Example usage::

        manager = SecretManager(prefix="secret:", keys=["OPENAI_API_KEY"])
        # Load secrets
        mapping = await manager.load_secrets(
            secret_port=keyvault,
            memory=memory,
            dag_id="my_pipeline"
        )
        # Clean up after execution
        await manager.cleanup_secrets(
            memory=memory,
            dag_id="my_pipeline"
        )
    """

    def __init__(
        self,
        secret_keys: list[str] | None = None,
        secret_prefix: str = "secret:",  # nosec B107 - Not a password, it's a key prefix
    ):
        """Initialize secret manager.

        Parameters
        ----------
        secret_keys : list[str] | None, default=None
            Specific secret keys to load. If None, loads all available secrets.
        secret_prefix : str, default="secret:"
            Prefix for secret keys in memory
        """
        self.secret_keys = secret_keys
        self.secret_prefix = secret_prefix
        self._loaded_secret_keys: dict[str, list[str]] = {}  # dag_id -> memory_keys

    async def load_secrets(
        self,
        secret_port: SecretStore | None,
        memory: Memory | None,
        dag_id: str,
    ) -> dict[str, str]:
        """Load secrets from SecretStore into Memory.

        Parameters
        ----------
        secret_port : SecretStore | None
            Secret port instance (KeyVault, etc.)
        memory : Memory | None
            Memory port instance to store secrets in
        dag_id : str
            DAG identifier for tracking

        Returns
        -------
        dict[str, str]
            Mapping of secret key → memory key

        Examples
        --------
        Example usage::

            mapping = await manager.load_secrets(
                secret_port=keyvault,
                memory=memory,
                dag_id="my_pipeline"
            )
            # Returns: {"OPENAI_API_KEY": "secret:OPENAI_API_KEY", ...}
        """
        if not secret_port:
            logger.debug("No secret port configured, skipping secret injection")
            return {}

        if not memory:
            logger.warning("Secret port configured but no memory port available")
            return {}

        try:
            # Load secrets into memory
            mapping = await secret_port.aload_secrets_to_memory(
                memory=memory, prefix=self.secret_prefix, keys=self.secret_keys
            )

            memory_keys = list(mapping.values())
            self._loaded_secret_keys[dag_id] = memory_keys

            logger.info(
                "Loaded {} secrets into memory with prefix '{}'",
                len(mapping),
                self.secret_prefix,
            )
            logger.debug("Secret keys loaded: {}", list(mapping.keys()))

            return mapping

        except (ValueError, KeyError, RuntimeError) as e:
            # Secret loading errors
            logger.error("Failed to inject secrets: {}", e, exc_info=True)
            raise

    async def cleanup_secrets(
        self,
        memory: Memory | None,
        dag_id: str,
    ) -> dict[str, Any]:
        """Remove secrets from Memory for security.

        Parameters
        ----------
        memory : Memory | None
            Memory port instance
        dag_id : str
            DAG identifier

        Returns
        -------
        dict[str, Any]
            Cleanup results with keys_removed count

        Examples
        --------
        Example usage::

            result = await manager.cleanup_secrets(
                memory=memory,
                dag_id="my_pipeline"
            )
            # {"cleaned": True, "keys_removed": 2}
        """
        if not memory:
            logger.debug("No memory port available for secret cleanup")
            return {"cleaned": False, "reason": "No memory port"}

        secret_keys = self.get_loaded_secret_keys(dag_id)

        if not secret_keys:
            logger.debug("No secrets were loaded for this pipeline")
            return {"cleaned": True, "keys_removed": 0}

        # Remove each secret from memory
        removed_count = 0
        for secret_key in secret_keys:
            try:
                await memory.aset(secret_key, None)
                removed_count += 1
                logger.debug("Removed secret from memory: {}", secret_key)
            except (RuntimeError, ValueError, KeyError) as e:
                # Secret removal errors - log but continue cleanup
                logger.warning("Failed to remove secret '{}': {}", secret_key, e)

        # Clean up tracked keys
        self.clear_loaded_secret_keys(dag_id)

        logger.info("Secret cleanup: Removed {} secret(s) from memory", removed_count)
        return {"cleaned": True, "keys_removed": removed_count}

    def get_loaded_secret_keys(self, dag_id: str) -> list[str]:
        """Get the list of secret keys loaded for a specific pipeline.

        Parameters
        ----------
        dag_id : str
            The DAG identifier

        Returns
        -------
        list[str]
            List of memory keys where secrets were stored
        """
        return self._loaded_secret_keys.get(dag_id, [])

    def clear_loaded_secret_keys(self, dag_id: str) -> None:
        """Clear the tracked secret keys for a specific pipeline.

        Parameters
        ----------
        dag_id : str
            The DAG identifier
        """
        if dag_id in self._loaded_secret_keys:
            del self._loaded_secret_keys[dag_id]
