"""Vulture whitelist for false positives.

This file contains imports and names that vulture incorrectly identifies as unused.
Each entry should have a comment explaining why it's needed.
"""

# Used in cast() string literal for type checking (line 467 in registry.py)
from collections.abc import Callable as CallableType

# Dummy usage to satisfy vulture
_ = CallableType


# Protocol abstract method parameters (intentionally unused - define interface)
# hexdag/core/ports/api_call.py - APICallPort protocol
def aget(url: str, headers: dict, params: dict, **kwargs: object) -> None:
    """Whitelist for APICallPort.aget parameters."""
    _ = (url, headers, params, kwargs)


def apost(url: str, json: dict, data: dict, headers: dict, **kwargs: object) -> None:
    """Whitelist for APICallPort.apost parameters."""
    _ = (url, json, data, headers, kwargs)


def aput(url: str, json: dict, data: dict, headers: dict, **kwargs: object) -> None:
    """Whitelist for APICallPort.aput parameters."""
    _ = (url, json, data, headers, kwargs)


def adelete(url: str, headers: dict, **kwargs: object) -> None:
    """Whitelist for APICallPort.adelete parameters."""
    _ = (url, headers, kwargs)


# hexdag/core/ports/file_storage.py - FileStoragePort protocol
def aupload(local_path: str, remote_path: str) -> None:
    """Whitelist for FileStoragePort.aupload parameters."""
    _ = (local_path, remote_path)


def adownload(remote_path: str, local_path: str) -> None:
    """Whitelist for FileStoragePort.adownload parameters."""
    _ = (remote_path, local_path)


def adelete_file(remote_path: str) -> None:
    """Whitelist for FileStoragePort.adelete parameter."""
    _ = remote_path


def aexists(remote_path: str) -> None:
    """Whitelist for FileStoragePort.aexists parameter."""
    _ = remote_path


def aget_metadata(remote_path: str) -> None:
    """Whitelist for FileStoragePort.aget_metadata parameter."""
    _ = remote_path
