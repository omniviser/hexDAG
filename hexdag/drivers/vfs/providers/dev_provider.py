"""VFS provider for /dev/ — bound port/adapter information.

Exposes which adapters are bound to which ports for the current
pipeline execution context.
"""

from __future__ import annotations

import json
from typing import Any

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError


class DevProvider:
    """VFS provider for /dev/ — bound port/adapter information."""

    def __init__(self, port_bindings: dict[str, Any]) -> None:
        self._bindings = port_bindings

    async def read(self, relative_path: str) -> str:
        """Read port binding details.

        Paths
        -----
        - ``ports/<name>`` → JSON with adapter class, config
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/dev/", "cannot read directory; use readdir")

        parts = path.split("/")
        if parts[0] != "ports":
            raise VFSError(f"/dev/{path}", "only /dev/ports/ is available")

        if len(parts) == 1:
            return json.dumps(list(self._bindings.keys()), indent=2)

        port_name = parts[1]
        if port_name not in self._bindings:
            raise VFSError(
                f"/dev/{path}",
                f"port '{port_name}' not bound. Available: {sorted(self._bindings.keys())}",
            )

        binding = self._bindings[port_name]
        info: dict[str, Any] = {}
        if isinstance(binding, dict):
            info = binding
        else:
            info = {
                "adapter_class": type(binding).__name__,
                "module_path": f"{type(binding).__module__}.{type(binding).__qualname__}",
            }
        return json.dumps(info, indent=2, default=str)

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List bound ports.

        Paths
        -----
        - ``""`` → ["ports"]
        - ``ports`` → list of bound port names
        """
        path = relative_path.strip("/")

        if not path:
            return [DirEntry(name="ports", entry_type=EntryType.DIRECTORY, path="/dev/ports")]

        if path == "ports":
            return [
                DirEntry(
                    name=name,
                    entry_type=EntryType.FILE,
                    path=f"/dev/ports/{name}",
                )
                for name in sorted(self._bindings.keys())
            ]

        raise VFSError(f"/dev/{path}", "not a directory")

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a port binding."""
        path = relative_path.strip("/")

        if not path:
            return StatResult(
                path="/dev",
                entry_type=EntryType.DIRECTORY,
                description="Device and port bindings",
                child_count=1,
                capabilities=["read"],
            )

        parts = path.split("/")
        if parts[0] == "ports" and len(parts) == 1:
            return StatResult(
                path="/dev/ports",
                entry_type=EntryType.DIRECTORY,
                description="Bound port adapters",
                child_count=len(self._bindings),
                capabilities=["read"],
            )

        if parts[0] == "ports" and len(parts) == 2:
            port_name = parts[1]
            if port_name not in self._bindings:
                raise VFSError(f"/dev/{path}", f"port '{port_name}' not bound")

            binding = self._bindings[port_name]
            adapter_class = type(binding).__name__ if not isinstance(binding, dict) else "dict"

            return StatResult(
                path=f"/dev/ports/{port_name}",
                entry_type=EntryType.FILE,
                entity_type="port",
                description=f"Port '{port_name}' bound to {adapter_class}",
                capabilities=["read"],
                tags={"port_name": port_name},
            )

        raise VFSError(f"/dev/{path}", "path not found")
