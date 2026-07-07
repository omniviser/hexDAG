"""Back-compat shim — the SQLAlchemy adapter moved to the database plugin.

``SQLAlchemyAdapter`` and ``SQLAlchemyStateBackend`` now live in
``hexdag_plugins.database`` (install with ``pip install hexdag-plugins[database]``).
Imports from this legacy path keep working when the plugin is installed;
otherwise a clear error points there.
"""

from typing import Any

_MOVED = {
    "SQLAlchemyAdapter": "hexdag_plugins.database.adapters",
    "SQLAlchemyStateBackend": "hexdag_plugins.database.adapters",
}

# Names resolved lazily via __getattr__ (PEP 562) from the database plugin.
__all__ = [
    "SQLAlchemyAdapter",  # noqa: F822 # pyright: ignore[reportUnsupportedDunderAll]
    "SQLAlchemyStateBackend",  # noqa: F822 # pyright: ignore[reportUnsupportedDunderAll]
]


def __getattr__(name: str) -> Any:  # PEP 562
    module_path = _MOVED.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import importlib

        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:  # pragma: no cover - depends on plugin install
        raise ImportError(
            f"{name} moved to hexdag_plugins.database; "
            f"install it with `pip install hexdag-plugins[database]`"
        ) from exc
