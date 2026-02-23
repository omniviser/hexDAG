"""Pipeline modules for hexDAG framework.

This package contains the YAML workflow builder for creating DirectedGraphs
from declarative YAML configurations, plus the configuration loader.

"""

# Import tags to register YAML custom tags
from . import include_tag as _include_tag  # noqa: F401
from . import py_tag as _py_tag  # noqa: F401
from .include_tag import set_include_base_path
from .tag_discovery import discover_tags, get_known_tag_names, get_tag_schema
from .yaml_builder import YamlPipelineBuilder


def __getattr__(name: str) -> object:
    """Lazy imports for config loader symbols to avoid circular imports."""
    _config_names = {
        "ConfigLoader",
        "clear_config_cache",
        "config_to_manifest_entries",
        "get_default_config",
        "load_config",
    }
    if name in _config_names:
        from hexdag.compiler import config_loader

        return getattr(config_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "YamlPipelineBuilder",
    "discover_tags",
    "get_known_tag_names",
    "get_tag_schema",
    "set_include_base_path",
]
