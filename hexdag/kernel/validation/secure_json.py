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
    "yaml_error",
]


logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for hot-path text processing
_RE_COMMENT = re.compile(r"(?m)\s*(//|#).*?$")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_RE_SINGLE_QUOTES = re.compile(r"(?<!\\)'([^']*?)'(?!\\)")
_RE_JSON_BLOCK = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
_RE_CODE_BLOCK = re.compile(r"```[\w-]*\s*([\s\S]*?)```")
_RE_RAW_JSON = re.compile(r"[\[{][\s\S]*[\]}]")
_RE_YAML_BLOCK = re.compile(r"```ya?ml\s*([\s\S]*?)```", re.IGNORECASE)
_RE_GENERIC_BLOCK = re.compile(r"```\s*([\s\S]*?)```")


@dataclass
class SafeJSONResult:
    data: Any | None = None
    error: ErrorCode | None = None
    message: str | None = None
    line: int | None = None
    col: int | None = None
    preview: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


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
            return SafeJSONResult(error="too_large", message="JSON exceeds size limit")

        if self._estimate_depth(text) > self.max_depth:
            return SafeJSONResult(error="too_deep", message="JSON exceeds depth limit")

        # Step 1: try orjson
        try:
            return SafeJSONResult(data=orjson.loads(text))
        except Exception as exc:
            logger.debug("orjson initial parse failed; attempting cleanup fallback: %s", exc)

        # Step 2: cleanup + retry
        # Note: only re-check size (cleanup can change byte count via comment removal).
        # Depth is NOT re-checked — _cleanup only does regex substitutions
        # (comments, trailing commas, quotes) which cannot change nesting depth.
        cleaned = self._cleanup(text)
        if len(cleaned.encode("utf-8")) <= self.max_size_bytes:
            try:
                return SafeJSONResult(data=orjson.loads(cleaned))
            except Exception as exc:  # noqa: BLE001 - we deliberately log and fall back
                logger.debug(
                    "orjson cleaned parse failed; falling back to stdlib for diagnostics: %s", exc
                )

        # Step 3: stdlib json for diagnostics
        try:
            return SafeJSONResult(data=json.loads(cleaned, parse_constant=lambda _: None))
        except json.JSONDecodeError as e:
            preview = self._format_error_line(cleaned, e.lineno, e.colno)
            return SafeJSONResult(
                error="invalid_syntax",
                message=e.msg,
                line=e.lineno,
                col=e.colno,
                preview=preview,
            )
        except Exception:
            return SafeJSONResult(error="unrecoverable", message="Unrecoverable JSON")

    def loads_from_text(self, text: str) -> SafeJSONResult:
        """Extract and parse JSON from mixed text (LLM output, markdown, etc.).

        Tries in order: ```json blocks → generic code blocks → raw JSON match.
        Then applies size/depth validation and cleanup.
        """
        candidate = self._extract_json(text)
        if not candidate:
            return SafeJSONResult(error="no_json_found", message="No JSON found in text")
        return self.loads(candidate)

    def loads_yaml(self, text: str) -> SafeJSONResult:
        """Parse YAML text with size validation.

        Uses ``yaml.safe_load`` for parsing. Only size is checked (YAML depth
        estimation is not straightforward), so callers relying on depth limits
        should prefer JSON where possible.
        """
        import yaml  # lazy: optional yaml dependency

        if len(text.encode("utf-8")) > self.max_size_bytes:
            return SafeJSONResult(error="too_large", message="YAML exceeds size limit")

        try:
            data = yaml.safe_load(text)
            return SafeJSONResult(data=data)
        except yaml.YAMLError as e:
            return SafeJSONResult(error="yaml_error", message=str(e))

    def loads_yaml_from_text(self, text: str) -> SafeJSONResult:
        """Extract YAML from markdown code blocks and parse it.

        Tries ```yaml blocks first, then generic code blocks, finally raw text.
        """
        candidate = self._extract_yaml(text)
        return self.loads_yaml(candidate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cleanup(text: str) -> str:
        text = _RE_COMMENT.sub("", text)
        text = _RE_TRAILING_COMMA.sub(r"\1", text)
        return _RE_SINGLE_QUOTES.sub(r'"\1"', text)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        if not text:
            return None
        match = _RE_JSON_BLOCK.search(text)
        if match:
            return match.group(1).strip()
        match = _RE_CODE_BLOCK.search(text)
        if match and match.group(1).lstrip().startswith(("{", "[")):
            return match.group(1).strip()
        match = _RE_RAW_JSON.search(text)
        return match.group(0).strip() if match else None

    @staticmethod
    def _extract_yaml(text: str) -> str:
        """Extract YAML content from markdown code blocks or return raw text."""
        if not text:
            return text
        match = _RE_YAML_BLOCK.search(text)
        if match:
            return match.group(1).strip()
        match = _RE_GENERIC_BLOCK.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

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
