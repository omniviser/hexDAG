"""Pipeline modules for hexDAG framework.

This package contains the YAML workflow builder for creating DirectedGraphs
from declarative YAML configurations.

"""

# Import tags to register YAML custom tags
from . import include_tag as _include_tag  # noqa: F401
from . import py_tag as _py_tag  # noqa: F401
from .include_tag import set_include_base_path
from .tag_discovery import discover_tags, get_known_tag_names, get_tag_schema
from .yaml_builder import YamlPipelineBuilder

__all__ = [
    "YamlPipelineBuilder",
    "set_include_base_path",
    "discover_tags",
    "get_known_tag_names",
    "get_tag_schema",
]
