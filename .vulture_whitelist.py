"""Vulture whitelist for false positives.

This file contains imports and names that vulture incorrectly identifies as unused.
Each entry should have a comment explaining why it's needed.
"""

# Used in cast() string literal for type checking (line 467 in registry.py)
from collections.abc import Callable as CallableType

# Dummy usage to satisfy vulture
_ = CallableType


# Protocol abstract method parameters (intentionally unused - define interface)
# hexdag/kernel/ports/api_call.py - ApiCall protocol
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


# hexdag/kernel/ports/file_storage.py - FileStorage protocol
def aupload(local_path: str, remote_path: str) -> None:
    """Whitelist for FileStorage.aupload parameters."""
    _ = (local_path, remote_path)


def adownload(remote_path: str, local_path: str) -> None:
    """Whitelist for FileStorage.adownload parameters."""
    _ = (remote_path, local_path)


def adelete_file(remote_path: str) -> None:
    """Whitelist for FileStorage.adelete parameter."""
    _ = remote_path


def aexists(remote_path: str) -> None:
    """Whitelist for FileStorage.aexists parameter."""
    _ = remote_path


def aget_metadata(remote_path: str) -> None:
    """Whitelist for FileStorage.aget_metadata parameter."""
    _ = remote_path


# hexdag/kernel/ports/data_store.py - SupportsTTL protocol
def aset_with_ttl(key: str, value: object, ttl_seconds: int) -> None:
    """Whitelist for SupportsTTL.aset_with_ttl parameters."""
    _ = (key, value, ttl_seconds)
