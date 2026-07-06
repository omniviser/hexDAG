"""Email adapters for hexDAG."""

from hexdag_plugins.mail.adapters.graph import GraphEmailAdapter
from hexdag_plugins.mail.adapters.noop import NoopEmailAdapter
from hexdag_plugins.mail.adapters.sendgrid import SendGridEmailAdapter

__all__ = [
    "GraphEmailAdapter",
    "NoopEmailAdapter",
    "SendGridEmailAdapter",
]
