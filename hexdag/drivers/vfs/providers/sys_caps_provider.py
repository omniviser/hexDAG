"""VFS provider for /sys/caps/ — capability introspection.

Exposes the current ``CapSet``, available capabilities taxonomy,
and named capability profiles from ``kind: Config``.

Paths
-----
- ``current``       — JSON of the active CapSet
- ``taxonomy``      — JSON of all known capabilities with descriptions
- ``profiles/``     — list of named cap profiles
- ``profiles/<name>`` — JSON of a specific named profile
"""

from __future__ import annotations

import json
from typing import Any

from hexdag.kernel.domain.caps import ALL_CAPABILITIES, CAPABILITY_DESCRIPTIONS, CapSet
from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError


class SysCapsProvider:
    """VFS provider for ``/sys/caps/`` — capability introspection.

    Parameters
    ----------
    cap_set : CapSet | None
        The active capability set. None means unrestricted.
    profiles : dict[str, dict[str, list[str]]] | None
        Named capability profiles from ``DefaultCaps.profiles``.
    """

    def __init__(
        self,
        cap_set: CapSet | None = None,
        profiles: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self._cap_set = cap_set or CapSet.unrestricted()
        self._profiles = profiles or {}

    def set_caps(self, cap_set: CapSet) -> None:
        """Update the active CapSet."""
        self._cap_set = cap_set

    def set_profiles(self, profiles: dict[str, dict[str, list[str]]]) -> None:
        """Update the profiles registry."""
        self._profiles = profiles

    async def read(self, relative_path: str) -> str:
        """Read capability data at a relative path."""
        path = relative_path.strip("/")

        if path == "current":
            return self._read_current()

        if path == "taxonomy":
            return self._read_taxonomy()

        if path.startswith("profiles/"):
            profile_name = path[len("profiles/") :]
            return self._read_profile(profile_name)

        raise VFSError(f"/sys/caps/{path}", "path not found")

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List entries in a directory."""
        path = relative_path.strip("/")

        if path == "" or path == "/":
            return [
                DirEntry(name="current", entry_type=EntryType.FILE, path="/sys/caps/current"),
                DirEntry(name="taxonomy", entry_type=EntryType.FILE, path="/sys/caps/taxonomy"),
                DirEntry(
                    name="profiles", entry_type=EntryType.DIRECTORY, path="/sys/caps/profiles"
                ),
            ]

        if path == "profiles":
            return [
                DirEntry(
                    name=name,
                    entry_type=EntryType.FILE,
                    path=f"/sys/caps/profiles/{name}",
                )
                for name in sorted(self._profiles)
            ]

        raise VFSError(f"/sys/caps/{path}", "not a directory")

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a path."""
        path = relative_path.strip("/")

        if path == "" or path == "/":
            return StatResult(
                path="/sys/caps",
                entry_type=EntryType.DIRECTORY,
                description="Capability introspection",
                child_count=3,
                capabilities=["read"],
            )

        if path == "current":
            return StatResult(
                path="/sys/caps/current",
                entry_type=EntryType.FILE,
                description="Active capability set (JSON)",
                capabilities=["read"],
            )

        if path == "taxonomy":
            return StatResult(
                path="/sys/caps/taxonomy",
                entry_type=EntryType.FILE,
                description="All known capabilities with descriptions (JSON)",
                capabilities=["read"],
            )

        if path == "profiles":
            return StatResult(
                path="/sys/caps/profiles",
                entry_type=EntryType.DIRECTORY,
                description="Named capability profiles",
                child_count=len(self._profiles),
                capabilities=["read"],
            )

        if path.startswith("profiles/"):
            profile_name = path[len("profiles/") :]
            if profile_name in self._profiles:
                return StatResult(
                    path=f"/sys/caps/profiles/{profile_name}",
                    entry_type=EntryType.FILE,
                    description=f"Capability profile '{profile_name}'",
                    capabilities=["read"],
                )

        raise VFSError(f"/sys/caps/{path}", "path not found")

    # --- Private helpers ---

    def _read_current(self) -> str:
        data: dict[str, Any] = {
            "allowed": sorted(self._cap_set.allowed),
            "denied": sorted(self._cap_set.denied),
        }
        return json.dumps(data, indent=2)

    def _read_taxonomy(self) -> str:
        taxonomy = [
            {"capability": cap, "description": CAPABILITY_DESCRIPTIONS.get(cap, "")}
            for cap in sorted(ALL_CAPABILITIES)
        ]
        return json.dumps(taxonomy, indent=2)

    def _read_profile(self, name: str) -> str:
        profile = self._profiles.get(name)
        if profile is None:
            available = sorted(self._profiles.keys())
            raise VFSError(
                f"/sys/caps/profiles/{name}",
                f"profile not found. Available: {available}",
            )
        return json.dumps(profile, indent=2)
