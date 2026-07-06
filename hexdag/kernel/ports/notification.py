"""DEPRECATED alias — the Notification port merged into the Messaging family.

``Notification`` is now :class:`hexdag.kernel.ports.messaging.SupportsNotification`,
a capability of the :class:`~hexdag.kernel.ports.messaging.Messaging` port
(alongside ``SupportsEmail`` for two-way threaded conversations). Existing
imports and ``isinstance`` checks keep working through this alias.

Use ``from hexdag.kernel.ports.messaging import SupportsNotification`` in
new code.
"""

from __future__ import annotations

from hexdag.kernel.ports.messaging import SupportsNotification

Notification = SupportsNotification

__all__ = ["Notification"]
