"""Email adapters for hexDAG."""

from hexdag_plugins.mail.adapters.graph import GraphEmailAdapter
from hexdag_plugins.mail.adapters.noop import NoopEmailAdapter

__all__ = [
    "GraphEmailAdapter",
    "NoopEmailAdapter",
]
