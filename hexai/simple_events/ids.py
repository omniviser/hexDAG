"""
Event ID generation utilities: uuid7 only.
"""


def generate_event_id(preferred: str | None = None) -> str:
    if preferred:
        return str(preferred)
    import uuid

    func = getattr(uuid, "uuid7", None)
    if func is None:
        raise RuntimeError("uuid.uuid7() required by simple events")
    return str(func())
