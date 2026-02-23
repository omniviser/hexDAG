"""Backward-compatibility re-exports from hexdag.compiler.config_loader.

.. deprecated::
    This module has moved to ``hexdag.compiler.config_loader``. Import from there instead.
    This re-export wrapper will be removed in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "hexdag.kernel.config.loader has moved to hexdag.compiler.config_loader. "
    "Update your imports to use 'from hexdag.compiler.config_loader import ...'.",
    DeprecationWarning,
    stacklevel=2,
)

from hexdag.compiler.config_loader import (  # noqa: E402, F401
    ConfigLoader,
    _parse_bool_env,
    clear_config_cache,
    config_to_manifest_entries,
    get_default_config,
    load_config,
)

__all__ = [
    "ConfigLoader",
    "_parse_bool_env",
    "clear_config_cache",
    "config_to_manifest_entries",
    "get_default_config",
    "load_config",
]
