"""
Event ID generation utilities: prefer uuid7, fallback to uuid4 if unavailable.
"""

import uuid
import warnings


def generate_event_id(preferred: str | None = None) -> str:
    if preferred:
        return str(preferred)

    func = getattr(uuid, "uuid7", None)
    if func is not None:
        return str(func())
    warnings.warn("uuid.uuid7() missing; falling back to uuid4 (dev only)", stacklevel=2)
    return str(uuid.uuid4())
