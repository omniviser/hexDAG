"""Backward-compatibility re-exports from hexdag.compiler.

.. deprecated::
    This module has moved to ``hexdag.compiler``. Import from there instead.
    This re-export wrapper will be removed in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "hexdag.kernel.pipeline_builder has moved to hexdag.compiler. "
    "Update your imports to use 'from hexdag.compiler import ...'.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from hexdag.compiler import (  # noqa: E402, F401
    YamlPipelineBuilder,
    discover_tags,
    get_known_tag_names,
    get_tag_schema,
    set_include_base_path,
)

__all__ = [
    "YamlPipelineBuilder",
    "set_include_base_path",
    "discover_tags",
    "get_known_tag_names",
    "get_tag_schema",
]
