"""Configuration loading and management for HexDAG."""

from hexdag.kernel.config.models import (
    DefaultCaps,
    DefaultLimits,
    HexDAGConfig,
    LoggingConfig,
    ManifestEntry,
)


def __getattr__(name: str) -> object:
    """Lazy imports for config loader symbols (moved to hexdag.compiler.config_loader)."""
    _loader_names = {
        "ConfigLoader",
        "clear_config_cache",
        "config_to_manifest_entries",
        "get_default_config",
        "load_config",
    }
    if name in _loader_names:
        from hexdag.compiler import config_loader

        return getattr(config_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DefaultCaps",
    "DefaultLimits",
    "HexDAGConfig",
    "LoggingConfig",
    "ManifestEntry",
]
