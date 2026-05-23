"""Email plugins for hexDAG."""

from hexdag_plugins.mail.adapters import (
    GraphEmailAdapter,
    NoopEmailAdapter,
)

__all__ = [
    "GraphEmailAdapter",
    "NoopEmailAdapter",
]
