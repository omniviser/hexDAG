"""Port protocols used by hexdag_plugins.

The email port family (``SupportsEmail``, ``SendEmailRequest``,
``SendEmailResult``, ``html_to_plain_text``) moved to
``hexdag.kernel.ports.messaging`` as part of the unified Messaging port —
this module re-exports it for backward compatibility.

``FileStorage`` still lives here (used only by the plugins package).
"""

from typing import Protocol, runtime_checkable

from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.ports.messaging import (  # noqa: F401  (back-compat re-exports)
    SendEmailRequest,
    SendEmailResult,
    SupportsEmail,
    _HTMLTextExtractor,
    html_to_plain_text,
)

__all__ = [
    "FileStorage",
    "SendEmailRequest",
    "SendEmailResult",
    "SupportsEmail",
    "html_to_plain_text",
]


@runtime_checkable
class FileStorage(Protocol):
    """Port for file storage operations.

    Provides a unified interface for local and cloud file storage.
    """

    async def aupload(self, local_path: str, remote_path: str) -> dict:
        """Upload a file."""
        ...

    async def adownload(self, remote_path: str, local_path: str) -> dict:
        """Download a file."""
        ...

    async def adelete(self, remote_path: str) -> dict:
        """Delete a file."""
        ...

    async def alist(self, prefix: str = "") -> list[str]:
        """List files with optional prefix."""
        ...

    async def aexists(self, remote_path: str) -> bool:
        """Check if file exists."""
        ...

    async def aget_metadata(self, remote_path: str) -> dict:
        """Get file metadata."""
        ...

    async def ahealth_check(self) -> HealthStatus:
        """Check storage health."""
        ...
