"""Component manifest system for declarative registration.

The manifest is the single source of truth for what components are installed.
Similar to Django's INSTALLED_APPS, it declares which modules contain components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False


@dataclass
class ManifestEntry:
    """Single entry in the component manifest."""

    namespace: str
    module: str

    def __post_init__(self) -> None:
        """Validate manifest entry."""
        if not self.namespace:
            raise ValueError("Namespace cannot be empty")
        if not self.module:
            raise ValueError("Module path cannot be empty")
        if ":" in self.namespace:
            raise ValueError(f"Namespace cannot contain ':' - got {self.namespace}")


class ComponentManifest:
    """Container for component manifest (structure only, no config)."""

    def __init__(self, entries: list[dict[str, str]] | list[ManifestEntry]):
        """Initialize manifest from list of entries.

        Parameters
        ----------
        entries : list[dict[str, str]] | list[ManifestEntry]
            List of manifest entries. Each entry must have 'namespace' and 'module'.

        Examples
        --------
        >>> manifest = ComponentManifest([
        ...     {"namespace": "core", "module": "hexai.core.nodes"},
        ...     {"namespace": "plugin", "module": "my_plugin.components"},
        ... ])
        """
        self.entries: list[ManifestEntry] = []
        self._namespace_to_modules: dict[str, list[str]] = {}

        for entry in entries:
            manifest_entry = ManifestEntry(**entry) if isinstance(entry, dict) else entry
            self.entries.append(manifest_entry)

            # Build namespace mapping
            if manifest_entry.namespace not in self._namespace_to_modules:
                self._namespace_to_modules[manifest_entry.namespace] = []
            self._namespace_to_modules[manifest_entry.namespace].append(manifest_entry.module)

    def get_modules_for_namespace(self, namespace: str) -> list[str]:
        """Get all modules that should be registered under a namespace."""
        return self._namespace_to_modules.get(namespace, [])

    def validate(self) -> None:
        """Validate manifest for duplicates and conflicts.

        Raises
        ------
        ValueError
            If there are duplicate (namespace, module) pairs.
        """
        seen = set()
        for entry in self.entries:
            key = (entry.namespace, entry.module)
            if key in seen:
                raise ValueError(
                    f"Duplicate manifest entry: namespace='{entry.namespace}', "
                    f"module='{entry.module}'"
                )
            seen.add(key)


def load_manifest_from_yaml(yaml_path: str | Path) -> ComponentManifest:
    """Load a component manifest from a YAML file.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML manifest file.

    Returns
    -------
    ComponentManifest
        The loaded manifest.

    Examples
    --------
    >>> manifest = load_manifest_from_yaml("hexai/core/component_manifest.yaml")
    >>> registry.bootstrap(manifest)
    """
    if not HAS_YAML or yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML manifests. Install it with: pip install pyyaml"
        )

    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data or "components" not in data:
        raise ValueError(f"Invalid manifest format in {yaml_path}: missing 'components' key")

    # Extract components list
    components = data["components"]

    # Create manifest (structure only)
    manifest = ComponentManifest(components)

    return manifest


def get_default_manifest() -> ComponentManifest:
    """Get the default HexDAG component manifest.

    This loads the standard manifest from hexai/core/component_manifest.yaml.

    Returns
    -------
    ComponentManifest
        The default manifest.
    """
    # Find the manifest file relative to this module
    manifest_path = Path(__file__).parent.parent / "component_manifest.yaml"

    if not manifest_path.exists():
        # Fallback to a minimal manifest if file doesn't exist
        return ComponentManifest(
            [
                {"namespace": "core", "module": "hexai.core.nodes"},
            ]
        )

    return load_manifest_from_yaml(manifest_path)
