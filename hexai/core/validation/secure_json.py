"""Secure JSON utilities with orjson backend and LLM-friendly helpers.

This module centralizes JSON parsing to provide consistent safety and
robustness characteristics across the codebase:

- Uses ``orjson`` for fast, strict JSON parsing
- Protects against JSON bombs via size and depth limits
- Provides minimal cleanup for common LLM formatting quirks
- Supports extracting JSON content from markdown/text outputs

Design principles:
- Fail safely with ``None`` rather than raising, unless explicitly requested
- Never attempt semantic corrections; only surface-level formatting fixes
- Keep helpers composable and readable
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import orjson

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE_BYTES = 1_000_000  # 1 MB
DEFAULT_MAX_DEPTH = 20


# ----------------------------
# Precompiled regexes (module)
# ----------------------------

# Fenced code blocks with explicit JSON label
FENCED_JSON_BLOCK_RE: re.Pattern[str] = re.compile(
    r"(?:^|\n)```+\s*json[^\n]*\n(?P<body>.*?)(?:\n)?```+",
    re.IGNORECASE | re.DOTALL,
)

# Any fenced code block (used if no explicitly-labeled JSON block found)
FENCED_ANY_BLOCK_RE: re.Pattern[str] = re.compile(
    r"(?:^|\n)```+[^\n]*\n(?P<body>.*?)(?:\n)?```+",
    re.DOTALL,
)

# Looks-like-JSON quick check
LOOKS_LIKE_JSON_RE: re.Pattern[str] = re.compile(r"^\s*[{\[]", re.DOTALL)

# Heuristic, non-balanced object/array grabbers capped at ~1 MiB
_HEURISTIC_CAP = DEFAULT_MAX_SIZE_BYTES
FIRST_OBJECT_BLOCK_RE: re.Pattern[str] = re.compile(
    rf"\{{[\s\S]{{0,{_HEURISTIC_CAP}}}?\}}", re.DOTALL
)
FIRST_ARRAY_BLOCK_RE: re.Pattern[str] = re.compile(rf"\[[\s\S]{{0,{_HEURISTIC_CAP}}}?\]", re.DOTALL)

# Cleanup regexes (string-aware)
# Match double-quoted strings, single-quoted strings, or inline comments.
_STRINGS_OR_COMMENTS_RE: re.Pattern[str] = re.compile(
    r'"(?:\\.|[^"\\])*"'  # double-quoted string
    r"|'(?:\\.|[^'\\])*'"  # single-quoted string
    r"|//[^\r\n]*"  # // comment until EOL
    r"|#[^\r\n]*"  # # comment until EOL
)

# Match double/single quoted strings or a trailing comma before } or ]
_STRINGS_OR_TRAILING_COMMA_RE: re.Pattern[str] = re.compile(
    r'"(?:\\.|[^"\\])*"'  # double-quoted string
    r"|'(?:\\.|[^'\\])*'"  # single-quoted string
    r"|,(?=\s*[}\]])"  # trailing comma before } or ]
)

# Convert `'key': 'val'` or `'key': <non-string>` to JSON-compliant form.
# We operate only when a pair clearly looks like an object item:
# prefix must be one of `{`, `[` or `,`
# and suffix must be followed by `,`, `}` or `]`.
SINGLE_QUOTE_KV_RE: re.Pattern[str] = re.compile(
    r"""
    (?P<prefix>[{\[,]\s*)                # start of an element within an object/array
    '                                     # opening single quote for key
    (?P<key>(?:\\'|[^'])*?)               # key content (allowing escaped single quotes)
    '                                     # closing single quote for key
    \s*:\s*                               # colon separator
    (?:                                   # value:
        '(?P<val_str>(?:\\'|[^'])*?)'     # single-quoted string value
        | (?P<val_other>[^,}\]]+)
    )
    (?P<suffix>\s*(?=[,}\]]))             # must be followed by comma or closing brace/bracket
    """,
    re.VERBOSE | re.DOTALL,
)


def loads(
    data: str | bytes | bytearray,
    *,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    allow_cleanup_retry: bool = True,
) -> Any | None:
    """Parse JSON safely using orjson with protections and minimal cleanup.

    Parameters
    ----------
    data : str | bytes | bytearray
        Input JSON content. Strings are treated as UTF-8 text.
    max_size_bytes : int
        Maximum allowed size in bytes to mitigate JSON bombs.
    max_depth : int
        Maximum allowed nesting depth for arrays/objects.
    allow_cleanup_retry : bool
        When True, perform minimal cleanup on common LLM formatting quirks and retry.

    Returns
    -------
    Any | None
        Parsed JSON value or ``None`` if parsing is unsafe or unrecoverable.
    """
    text, size_bytes = _coerce_to_text_and_size(data)
    if size_bytes > max_size_bytes:
        return None

    # First attempt: strict parse
    try:
        obj = orjson.loads(text)
    except orjson.JSONDecodeError:
        obj = None

    # Optional cleanup & retry (regex-only)
    if obj is None and allow_cleanup_retry:
        cleaned = _cleanup_json_minimal(text)
        if len(cleaned.encode("utf-8", errors="replace")) > max_size_bytes:
            return None
        try:
            obj = orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            return None

    if obj is None:
        return None

    # Enforce depth AFTER parsing by traversing the Python object (allowed to iterate here)
    if _max_container_depth(obj) > max_depth:
        return None

    return obj


def loads_from_llm_output(
    text: str,
    *,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    expected_type: type | tuple[type, ...] | str | None = None,
) -> Any | None:
    """Extract and parse JSON from an LLM-style text/markdown response.

    This helper first extracts likely JSON content from code fences or heuristic
    object/array regions (regex-only), then parses it with :func:`loads` including
    safety limits and cleanup retry. Optionally validates the parsed value's type.
    """
    candidate = extract_json_from_text(text)
    if candidate is None:
        return None

    parsed = loads(candidate, max_size_bytes=max_size_bytes, max_depth=max_depth)
    if parsed is None:
        return None

    target_type = _normalize_expected_type(expected_type)
    if target_type is not None and not isinstance(parsed, target_type):
        return None

    return parsed


def extract_json_from_text(text: str) -> str | None:
    """Extract a JSON substring from text, aware of markdown fenced code blocks.

    Extraction strategy (in order):
    1) Prefer fenced blocks marked as ```json
    2) Any fenced code block that "looks like" JSON
    3) Heuristic first `{...}` or `[...]` block (no balancing), capped at ~1 MiB
       — actual validity is determined by the parser downstream.
    """
    if not text:
        return None

    # 1) Prefer ```json fenced blocks
    m_json = FENCED_JSON_BLOCK_RE.search(text)
    if m_json:
        body = (m_json.group("body") or "").strip()
        if body:
            return body

    # 2) Any fenced block that looks like JSON
    for m in FENCED_ANY_BLOCK_RE.finditer(text):
        body = (m.group("body") or "").strip()
        if body and LOOKS_LIKE_JSON_RE.match(body):
            return body

    # 3) Heuristic first {...} or [...] without balancing, capped size
    m_obj = FIRST_OBJECT_BLOCK_RE.search(text)
    m_arr = FIRST_ARRAY_BLOCK_RE.search(text)

    first = None
    if m_obj and m_arr:
        first = m_obj if m_obj.start() < m_arr.start() else m_arr
    else:
        first = m_obj or m_arr

    if first:
        candidate = first.group(0).strip()
        # Hard cap at ~1 MiB worth of characters (approximate to bytes)
        if len(candidate) > DEFAULT_MAX_SIZE_BYTES:
            candidate = candidate[:DEFAULT_MAX_SIZE_BYTES]
        return candidate

    return None


ErrorCode = Literal[
    "too_large",
    "too_deep",
    "invalid_syntax",
    "unrecoverable",
    "no_json_found",
]


@dataclass
class SafeJSONResult:
    """Result container for safe JSON operations.

    Attributes
    ----------
    ok : bool
        Whether the operation succeeded.
    data : Any | None
        Parsed JSON value if successful.
    error : ErrorCode | None
        Machine-friendly error code when not successful.
    message : str | None
        Human-friendly error message when available.
    line_no : int | None
        1-based line number for syntax errors.
    col_no : int | None
        1-based column number for syntax errors.
    preview : str | None
        A short preview line with a caret position for diagnostics.
    """

    ok: bool
    data: Any | None = None
    error: ErrorCode | None = None
    message: str | None = None
    line_no: int | None = None
    col_no: int | None = None
    preview: str | None = None


class SafeJSON:
    """Stateful helper providing safe JSON parsing and extraction.

    This class adds pre-parse size/depth checks, minimal cleanup/retry,
    and helpful diagnostics suitable for LLM-oriented outputs.
    """

    def __init__(
        self, max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES, max_depth: int = DEFAULT_MAX_DEPTH
    ):
        self.max_size_bytes = max_size_bytes
        self.max_depth = max_depth

    def loads(self, data: str | bytes | bytearray) -> SafeJSONResult:
        """Parse JSON safely with protections and diagnostics.

        Parameters
        ----------
        data : str | bytes | bytearray
            Input JSON buffer.

        Returns
        -------
        SafeJSONResult
            Structured result with either parsed data or error details.
        """
        text, size_bytes = _coerce_to_text_and_size(data)

        if size_bytes > self.max_size_bytes:
            return SafeJSONResult(False, error="too_large", message="JSON exceeds size limit")

        if self._estimate_depth(text) > self.max_depth:
            return SafeJSONResult(False, error="too_deep", message="JSON exceeds depth limit")

        # Step 1: try orjson
        try:
            obj = orjson.loads(text)
            if _max_container_depth(obj) > self.max_depth:
                return SafeJSONResult(
                    False, error="too_deep", message="JSON exceeds depth limit after parsing"
                )
            return SafeJSONResult(True, data=obj)
        except orjson.JSONDecodeError as exc:
            logger.debug("orjson failed to parse input: %s", exc)

        # Step 2: cleanup + retry
        cleaned = _cleanup_json_minimal(text)
        cleaned_size = len(cleaned.encode("utf-8", errors="replace"))
        if cleaned_size <= self.max_size_bytes and self._estimate_depth(cleaned) <= self.max_depth:
            try:
                obj = orjson.loads(cleaned)
                if _max_container_depth(obj) > self.max_depth:
                    return SafeJSONResult(
                        False, error="too_deep", message="JSON exceeds depth limit after parsing"
                    )
                return SafeJSONResult(True, data=obj)
            except orjson.JSONDecodeError as exc:
                logger.debug("orjson failed to parse cleaned input: %s", exc)

        # Step 3: stdlib json for diagnostics
        try:
            obj = json.loads(cleaned, parse_constant=lambda _: None)
            if _max_container_depth(obj) > self.max_depth:
                return SafeJSONResult(
                    False, error="too_deep", message="JSON exceeds depth limit after parsing"
                )
            return SafeJSONResult(True, data=obj)
        except json.JSONDecodeError as e:
            preview = self._format_error_line(cleaned, e.lineno, e.colno)
            return SafeJSONResult(
                False,
                error="invalid_syntax",
                message=e.msg,
                line_no=e.lineno,
                col_no=e.colno,
                preview=preview,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            logger.debug("stdlib json raised non-decode error: %s", exc)
            return SafeJSONResult(False, error="unrecoverable", message="Unrecoverable JSON")

    def loads_from_text(self, text: str) -> SafeJSONResult:
        """Extract likely JSON from text/markdown and parse safely.

        Returns a structured result, including a helpful error code when
        no JSON-like content is found.
        """
        candidate = extract_json_from_text(text)
        if not candidate:
            return SafeJSONResult(False, error="no_json_found", message="No JSON found in text")
        return self.loads(candidate)

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
                if depth > max_depth:
                    max_depth = depth
            elif ch in "}]":
                depth = max(depth - 1, 0)
        return max_depth

    @staticmethod
    def _format_error_line(text: str, line_no: int, col_no: int) -> str | None:
        lines = text.splitlines()
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
            caret_line = " " * (col_no - 1) + "^"
            return f"{line}\n{caret_line}"
        return None


__all__ = [
    "DEFAULT_MAX_SIZE_BYTES",
    "DEFAULT_MAX_DEPTH",
    "loads",
    "loads_from_llm_output",
    "extract_json_from_text",
    "ErrorCode",
    "SafeJSONResult",
    "SafeJSON",
]


# Internal helpers


def _coerce_to_text_and_size(data: str | bytes | bytearray) -> tuple[str, int]:
    if isinstance(data, (bytes, bytearray)):
        b = bytes(data)
        return (b.decode("utf-8", errors="replace"), len(b))
    # For str, compute UTF-8 encoded size
    s = data
    return (s, len(s.encode("utf-8", errors="replace")))


def _cleanup_json_minimal(text: str) -> str:
    """Apply minimal, structure-preserving cleanup for typical LLM JSON quirks.

    Passes (regex-only):
      1) Remove full-line comments beginning with `//` or `#`.
      2) Remove trailing commas immediately before `}` or `]`.
      3) Convert `'key': 'val'` / `'key': <non-string>` pairs to JSON-compliant form.
         (performed iteratively up to 5 times; idempotent when stable)
    """
    s = _strip_inline_comments_preserving_strings(text)
    s = _remove_trailing_commas_outside_strings(s)

    def _kv_repl(m: re.Match[str]) -> str:
        key_raw = m.group("key")
        val_str = m.group("val_str")
        val_other = m.group("val_other")
        prefix = m.group("prefix")
        suffix = m.group("suffix") or ""

        key = _squoted_content_to_json_escaped(key_raw)
        if val_str is not None:
            val = _squoted_content_to_json_escaped(val_str)
            val_rendered = f'"{val}"'
        else:
            val_rendered = val_other.strip()

        return f'{prefix}"{key}": {val_rendered}{suffix}'

    # Run a few passes to cover nested/simple repeated patterns
    for _ in range(5):
        new_s = SINGLE_QUOTE_KV_RE.sub(_kv_repl, s)
        if new_s == s:
            break
        s = new_s

    return s


def _squoted_content_to_json_escaped(content: str) -> str:
    r"""Convert single-quoted content to a JSON-safe double-quoted form (payload only).

    - Unescape `\'` to `'`.
    - Escape bare `"` characters.
    - Leave other backslash escapes as-is.
    """
    s = re.sub(r"\\'", "'", content)
    s = re.sub(r'(?<!\\)"', r'\\"', s)
    return s


def _strip_inline_comments_preserving_strings(text: str) -> str:
    """Remove // and # comments while preserving quoted strings.

    Uses a tokenizer-style regex that matches either strings or comments.
    Strings are returned unchanged; comments are removed.
    """

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith('"') or token.startswith("'"):
            return token
        return ""

    return _STRINGS_OR_COMMENTS_RE.sub(_repl, text)


def _remove_trailing_commas_outside_strings(text: str) -> str:
    """Remove commas that appear immediately before } or ] outside of strings."""

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(","):
            return ""
        return token

    return _STRINGS_OR_TRAILING_COMMA_RE.sub(_repl, text)


def _max_container_depth(obj: Any) -> int:
    """Compute maximum nesting depth across dict/list containers only.

    The root container contributes depth 1. Scalars do not increase depth.
    Uses an explicit stack (iterative) to avoid Python recursion limits.
    """
    max_d = 0
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        node, d = stack.pop()
        if isinstance(node, dict):
            nd = d + 1
            max_d = max(max_d, nd)
            for v in node.values():
                stack.append((v, nd))
        elif isinstance(node, list):
            nd = d + 1
            max_d = max(max_d, nd)
            for v in node:
                stack.append((v, nd))
        # Scalars do not change depth
    return max_d


def _normalize_expected_type(
    expected: type | tuple[type, ...] | str | None,
) -> type | tuple[type, ...] | None:
    """Map a human-friendly expected_type to Python types.

    Supports strings: "object" -> dict, "array" -> list.
    Returns a type/tuple or None if no expectation is provided.
    """
    if expected is None:
        return None
    if isinstance(expected, str):
        name = expected.strip().lower()
        if name == "object":
            return dict
        if name == "array":
            return list
        return None
    return expected
