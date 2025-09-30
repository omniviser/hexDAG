"""Bootstrap lifecycle management."""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    InvalidComponentError,
    NamespacePermissionError,
    RegistryAlreadyBootstrappedError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexai.core.config import ManifestEntry
    from hexai.core.registry.component_store import ComponentStore

logger = logging.getLogger(__name__)


class BootstrapManager:
    """Manages registry bootstrap lifecycle.

    This class handles:
    - Bootstrap state machine (empty -> bootstrapping -> ready)
    - Manifest validation and loading
    - Module loading and error handling
    - Development vs production mode
    """

    def __init__(self, store: ComponentStore) -> None:
        """Initialize bootstrap manager.

        Args
        ----
        store : ComponentStore
            Component store to populate.
        """
        self._store = store
        self._ready = False
        self._manifest: list[ManifestEntry] | None = None
        self._dev_mode = False
        self._bootstrap_context = False

    @property
    def ready(self) -> bool:
        """Check if bootstrap is complete."""
        return self._ready

    @property
    def manifest(self) -> list[ManifestEntry] | None:
        """Get the current manifest."""
        return self._manifest

    @property
    def dev_mode(self) -> bool:
        """Check if in development mode."""
        return self._dev_mode

    @property
    def in_bootstrap_context(self) -> bool:
        """Check if currently bootstrapping."""
        return self._bootstrap_context

    def can_register(self) -> bool:
        """Check if registration is currently allowed.

        Returns
        -------
        bool
            True if registration is allowed.
        """
        # Can register if:
        # 1. Not ready yet (before bootstrap)
        # 2. In dev mode (after bootstrap but mutable)
        # 3. During bootstrap process
        return not self._ready or self._dev_mode or self._bootstrap_context

    def bootstrap(
        self,
        manifest: list[ManifestEntry],
        dev_mode: bool,
        register_components_fn: Callable[[object, str, str], int],
    ) -> None:
        """Bootstrap the registry from a manifest.

        Args
        ----
        manifest : list[ManifestEntry]
            Component manifest declaring what to load.
        dev_mode : bool
            If True, allows post-bootstrap registration.
        register_components_fn : Callable
            Function to call for each module to register its components.
            Signature: (registry, namespace, module_path) -> count

        """
        # Prepare for bootstrap
        manifest = self._prepare_bootstrap(manifest, dev_mode)

        # Load all modules from manifest
        self._bootstrap_context = True
        try:
            total_registered = self._load_manifest_modules(manifest, register_components_fn)
            self._finalize_bootstrap(total_registered, dev_mode)
        except Exception:
            # Clean up on failure
            self._cleanup_state()
            raise
        finally:
            self._bootstrap_context = False

    def _prepare_bootstrap(
        self,
        manifest: list[ManifestEntry],
        dev_mode: bool,
    ) -> list[ManifestEntry]:
        """Prepare for bootstrap and validate manifest.

        Args
        ----
        manifest : list[ManifestEntry]
            Manifest to validate.
        dev_mode : bool
            Development mode flag.

        Returns
        -------
        list[ManifestEntry]
            Validated manifest.

        Raises
        ------
        RegistryAlreadyBootstrappedError
            If already bootstrapped.
        ValueError
            If manifest has duplicates.
        """
        if self._ready:
            raise RegistryAlreadyBootstrappedError(
                "Registry has already been bootstrapped. Use reset() if you need to re-bootstrap."
            )

        # Validate entries for duplicates
        seen = set()
        for entry in manifest:
            key = (entry.namespace, entry.module)
            if key in seen:
                raise ValueError(
                    f"Duplicate manifest entry: namespace='{entry.namespace}', "
                    f"module='{entry.module}'"
                )
            seen.add(key)

        # Store configuration
        self._manifest = manifest
        self._dev_mode = dev_mode

        logger.info("Bootstrapping registry with %d entries", len(manifest))
        return manifest

    def _load_manifest_modules(
        self,
        manifest: list[ManifestEntry],
        register_components_fn: Callable[[object, str, str], int],
    ) -> int:
        """Load and register components from manifest modules.

        Args
        ----
        manifest : list[ManifestEntry]
            Manifest entries to load.
        register_components_fn : Callable
            Function to call for component registration.

        Returns
        -------
        int
            Total number of components registered.

        Raises
        ------
        ImportError
            If core module cannot be imported.
        ComponentAlreadyRegisteredError
            If component is already registered.
        InvalidComponentError
            If component is invalid.
        NamespacePermissionError
            If namespace permission denied.
        """
        total_registered = 0

        for entry in manifest:
            is_core_module = self._is_core_module(entry)

            # For non-core modules, check if they exist
            if not is_core_module:
                skip_reason = self._check_plugin_requirements(entry.module)
                if skip_reason:
                    logger.info(f"Skipping optional module {entry.module}: {skip_reason}")
                    continue

            try:
                count = register_components_fn(None, entry.namespace, entry.module)
                total_registered += count
                logger.info(
                    f"Registered {count} components from {entry.module} "
                    f"into namespace '{entry.namespace}'"
                )
            except ImportError as e:
                if not is_core_module:
                    logger.warning(f"Optional module {entry.module} not available: {e}")
                else:
                    logger.error("Failed to import core module %s: %s", entry.module, e)
                    raise
            except (
                ComponentAlreadyRegisteredError,
                InvalidComponentError,
                NamespacePermissionError,
            ) as e:
                logger.error("Failed to register components from %s: %s", entry.module, e)
                raise

        return total_registered

    def _is_core_module(self, entry: ManifestEntry) -> bool:
        """Check if manifest entry is a core module.

        Core modules must load successfully.

        Args
        ----
        entry : ManifestEntry
            Manifest entry to check.

        Returns
        -------
        bool
            True if core module.
        """
        return (
            entry.namespace == "core"
            or entry.module.startswith("hexai.core.")
            or entry.module == "hexai.tools.builtin_tools"
        )

    def _check_plugin_requirements(self, module_path: str) -> str | None:
        """Check if a plugin module exists.

        Args
        ----
        module_path : str
            Module path to check.

        Returns
        -------
        str | None
            Reason for skipping, or None if module exists.
        """
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return f"Module {module_path} not found"
        except (ModuleNotFoundError, ValueError) as e:
            return f"Module {module_path} not found: {e}"

        return None

    def _finalize_bootstrap(self, total_registered: int, dev_mode: bool) -> None:
        """Mark bootstrap as complete.

        Args
        ----
        total_registered : int
            Total components registered.
        dev_mode : bool
            Development mode flag.
        """
        self._ready = True
        logger.info(
            f"Bootstrap complete: {total_registered} components registered. "
            f"Registry is {'mutable (dev mode)' if dev_mode else 'read-only'}."
        )

    def _cleanup_state(self) -> None:
        """Clean up state on bootstrap failure."""
        self._store.clear()
        self._ready = False
        self._manifest = None
        self._bootstrap_context = False

    def reset(self) -> None:
        """Reset bootstrap state (for testing)."""
        self._ready = False
        self._manifest = None
        self._dev_mode = False
        self._bootstrap_context = False
