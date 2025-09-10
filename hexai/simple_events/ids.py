"""
Event ID generation utilities.

Prefer ULID if the optional 'ulid' package is available.
Fallback to UUIDv7 when supported by the runtime.
"""

from typing import Optional


def _generate_ulid() -> Optional[str]:
    """Return ULID string if 'ulid' package is available, otherwise None."""
    try:
        import importlib

        mod = importlib.import_module("ulid")  # optional dependency
        ulid_cls = getattr(mod, "ULID", None)
        if ulid_cls is None:
            return None
        return str(ulid_cls())

    except ModuleNotFoundError:
        return None
    except Exception:
        # Do not fail ID generation if optional backend misbehaves.
        return None


def _generate_uuidv7() -> Optional[str]:
    """Return UUIDv7 string if supported by this Python, otherwise None."""
    import uuid

    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())
    return None


def generate_event_id(preferred: Optional[str] = None) -> str:
    """
    Return provided ID or generate a ULID/UUIDv7 string.

    Parameters
    ----------
    preferred : Optional[str]
        If given, returned as-is (stringified).

    Returns
    -------
    str
        ULID when 'ulid' is installed, otherwise UUIDv7 if available.

    Raises
    ------
    RuntimeError
        When neither ULID nor UUIDv7 can be produced.
    """
    if preferred:
        return str(preferred)

    eid = _generate_ulid()
    if eid:
        return eid

    eid = _generate_uuidv7()
    if eid:
        return eid

    raise RuntimeError(
        "No ULID/UUIDv7 available. Install optional 'ulid' package or use Python with uuid.uuid7()."
    )
