import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import orjson

ErrorCode = Literal[
    "too_large",
    "too_deep",
    "invalid_syntax",
    "unrecoverable",
    "no_json_found",
]


logger = logging.getLogger(__name__)


@dataclass
class SafeJSONResult:
    ok: bool
    data: Any | None = None
    error: ErrorCode | None = None
    message: str | None = None
    line: int | None = None
    col: int | None = None
    preview: str | None = None


class SafeJSON:
    def __init__(self, max_size_bytes: int = 1_000_000, max_depth: int = 20):
        self.max_size_bytes = max_size_bytes
        self.max_depth = max_depth

    def loads(self, data: str | bytes | bytearray) -> SafeJSONResult:
        text = (
            data.decode("utf-8", errors="strict")
            if isinstance(data, (bytes, bytearray))
            else str(data)
        )

        if len(text.encode("utf-8")) > self.max_size_bytes:
            return SafeJSONResult(False, error="too_large", message="JSON exceeds size limit")

        if self._estimate_depth(text) > self.max_depth:
            return SafeJSONResult(False, error="too_deep", message="JSON exceeds depth limit")

        # Step 1: try orjson
        try:
            return SafeJSONResult(True, data=orjson.loads(text))
        except Exception as exc:
            logger.debug("orjson initial parse failed; attempting cleanup fallback: %s", exc)

        # Step 2: cleanup + retry
        cleaned = self._cleanup(text)
        if (
            len(cleaned.encode("utf-8")) <= self.max_size_bytes
            and self._estimate_depth(cleaned) <= self.max_depth
        ):
            try:
                return SafeJSONResult(True, data=orjson.loads(cleaned))
            except Exception as exc:  # noqa: BLE001 - we deliberately log and fall back
                logger.debug(
                    "orjson cleaned parse failed; falling back to stdlib for diagnostics: %s", exc
                )

        # Step 3: stdlib json for diagnostics
        try:
            return SafeJSONResult(True, data=json.loads(cleaned, parse_constant=lambda _: None))
        except json.JSONDecodeError as e:
            preview = self._format_error_line(cleaned, e.lineno, e.colno)
            return SafeJSONResult(
                False,
                error="invalid_syntax",
                message=e.msg,
                line=e.lineno,
                col=e.colno,
                preview=preview,
            )
        except Exception:
            return SafeJSONResult(False, error="unrecoverable", message="Unrecoverable JSON")

    def loads_from_text(self, text: str) -> SafeJSONResult:
        candidate = self._extract_json(text)
        if not candidate:
            return SafeJSONResult(False, error="no_json_found", message="No JSON found in text")
        return self.loads(candidate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cleanup(text: str) -> str:
        text = re.sub(r"(?m)\s*(//|#).*?$", "", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r"(?<!\\)'([^']*?)'(?!\\)", r'"\1"', text)
        return text

    @staticmethod
    def _extract_json(text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```[\w-]*\s*([\s\S]*?)```", text)
        if match and match.group(1).lstrip().startswith(("{", "[")):
            return match.group(1).strip()
        match = re.search(r"[\[{][\s\S]*[\]}]", text)
        return match.group(0).strip() if match else None

    @staticmethod
    def _estimate_depth(text: str) -> int:
        depth = 0
        max_depth = 0
        in_str: str | None = None
        esc = False
        for ch in text:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
            elif ch in "{[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch in "]}":
                depth = max(depth - 1, 0)
        return max_depth

    @staticmethod
    def _format_error_line(text: str, line_no: int, col_no: int, context: int = 1) -> str | None:
        """Return a snippet of the line with a caret pointing at the error col."""
        lines = text.splitlines()
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
            caret_line = " " * (col_no - 1) + "^"
            return f"{line}\n{caret_line}"
        return None
