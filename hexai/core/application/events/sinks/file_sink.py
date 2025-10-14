from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO, cast, runtime_checkable

from hexai.core.application.events.events import EVENT_REGISTRY
from hexai.core.ports.observer_manager import Observer


class FileSinkError(Exception):
    """Raised when the FileSinkObserver encounters an unrecoverable I/O error."""


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, Any]


def _event_to_dict(ev: Any) -> dict[str, Any]:
    """
    Convert an event dataclass instance into a JSON-serializable dictionary.
    """
    if dataclasses.is_dataclass(ev) and not isinstance(ev, type):
        # We know 'ev' is a dataclass instance here; help static checkers with a narrow cast.
        data: dict[str, Any] = dataclasses.asdict(cast("Any", ev))
    else:
        data = dict(getattr(ev, "__dict__", {}))

    ts = data.get("timestamp")
    if isinstance(ts, datetime):
        data["timestamp"] = ts.isoformat()

    cls = type(ev)
    spec = EVENT_REGISTRY.get(cls.__name__)
    if spec:
        data["type"] = spec.event_type
        envelope: dict[str, Any] = {}
        for out_key, src_attr in (spec.envelope_fields or {}).items():
            if src_attr in data:
                envelope[out_key] = data[src_attr]
        data["envelope"] = envelope

    return data


class FileSinkObserver(Observer):
    """
    Observer that writes every event as a single JSON line (JSONL).

    Characteristics:
    - Line-buffered writes plus explicit flush to ensure real-time tailing.
    - Produces well-formed UTF-8 JSON per line for easy parsing with jq.

    Fault isolation:
    - By default, handle() swallows I/O errors to avoid interrupting the pipeline.
    - Set raise_on_error=True to raise FileSinkError from handle() for strict testing.
    """

    def __init__(self, path: str, line_buffered: bool = True, raise_on_error: bool = False) -> None:
        """
        Initialize the file sink.

        Args:
            path: Target file path for JSONL output.
            line_buffered: If True, enable line buffering to flush on newline.
            raise_on_error: If True, handle() will raise FileSinkError on write/flush errors.
        """
        self._raise_on_error: bool = raise_on_error
        buffering = 1 if line_buffered else -1
        try:
            self._path = path
            self._f: TextIO = Path(self._path).open(  # noqa: SIM115, PTH123
                "a",
                buffering=buffering,
                encoding="utf-8",
            )
        except (OSError, ValueError) as e:
            raise FileSinkError(f"Failed to open file sink at '{path}': {e}") from e

    async def handle(self, event: Any) -> None:
        """
        Serialize and append the event as a JSON line, then flush.

        Raises:
            FileSinkError: only if raise_on_error=True and a write/flush error occurs.
        """
        obj = _event_to_dict(event)
        line = json.dumps(obj, ensure_ascii=False)

        try:
            self._f.write(line + "\n")
            self._f.flush()
        except (OSError, ValueError) as e:
            if self._raise_on_error:
                raise FileSinkError(f"Failed to write/flush to '{self._path}': {e}") from e
            # Fault isolation by default: do nothing to avoid breaking the pipeline.

    def close(self) -> None:
        """Close the underlying file handle safely; raise on explicit close failure."""
        try:
            self._f.close()
        except (OSError, ValueError) as e:
            # It’s reasonable to raise here because close() is a lifecycle action,
            # but if you prefer silent shutdowns, convert to a log instead.
            raise FileSinkError(f"Failed to close file sink '{self._path}': {e}") from e
