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

import re
from typing import Any, Iterable

import orjson

# Defaults chosen to balance safety and practicality
DEFAULT_MAX_SIZE_BYTES = 1_000_000
DEFAULT_MAX_DEPTH = 20


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
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8", errors="replace")
        size_bytes = len(data)
    else:
        text = data
        size_bytes = len(text.encode("utf-8", errors="replace"))

    if size_bytes > max_size_bytes:
        return None

    if _estimate_max_nesting_depth(text) > max_depth:
        return None

    try:
        return orjson.loads(text)
    except Exception:
        if not allow_cleanup_retry:
            return None

    # Minimal cleanup and second attempt
    cleaned = _cleanup_json_string(text)

    if len(cleaned.encode("utf-8", errors="replace")) > max_size_bytes:
        return None
    if _estimate_max_nesting_depth(cleaned) > max_depth:
        return None

    try:
        return orjson.loads(cleaned)
    except Exception:
        return None


def loads_from_llm_output(
    text: str,
    *,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    expected_type: type | tuple[type, ...] | str | None = None,
) -> Any | None:
    """Extract and parse JSON from an LLM-style text/markdown response.

    This helper first extracts likely JSON content from code fences or balanced
    brackets, then parses it with :func:`loads` including safety limits and
    cleanup retry. Optionally validates the parsed value's type.
    """
    extracted = extract_json_from_text(text)
    if extracted is None:
        return None
    parsed = loads(extracted, max_size_bytes=max_size_bytes, max_depth=max_depth)
    if parsed is None:
        return None
    target_type = _normalize_expected_type(expected_type)
    if target_type is not None:
        if isinstance(target_type, tuple):
            if not isinstance(parsed, target_type):
                return None
        else:
            if not isinstance(parsed, target_type):
                return None
    return parsed


def extract_json_from_text(text: str) -> str | None:
    """Extract JSON substring from text, aware of markdown fenced code blocks.

    Extraction strategy (in order):
    - Prefer fenced blocks marked as `````json```
    - Any fenced code block `````...```
    - Bracket matching starting from the first ``{`` or ``[``
    """
    if not text:
        return None

    # 1) Prefer ```json fenced blocks
    for lang, body in _iter_fenced_blocks(text):
        if lang and lang.lower() == "json":
            candidate = body.strip()
            if candidate:
                return candidate

    # 2) Any fenced block
    for _lang, body in _iter_fenced_blocks(text):
        candidate = body.strip()
        if candidate:
            # Only accept blocks that plausibly look like JSON
            if candidate.lstrip().startswith(("{", "[")):
                return candidate

    # 3) Bracket matching in free text
    return _extract_by_bracket_matching(text)


# Internal helpers


def _iter_fenced_blocks(text: str) -> Iterable[tuple[str | None, str]]:
    r"""Yield (language, body) for markdown fenced code blocks.

    Recognizes patterns like:
        ```json\n{...}\n```
        ```\n{...}\n```
    """
    # Non-greedy to capture the smallest matching block
    pattern = re.compile(r"```\s*([a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)\n?```", re.MULTILINE)
    for match in pattern.finditer(text):
        lang = match.group(1)
        body = match.group(2)
        yield (lang, body)


def _extract_by_bracket_matching(text: str) -> str | None:
    """Extract JSON by locating the first balanced object/array region."""
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None

    stack: list[str] = []
    string_quote: str | None = None
    escaping = False

    for i, ch in enumerate(text[start:], start=start):
        if string_quote is not None:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == string_quote:
                string_quote = None
        else:
            if ch in ('"', "'"):
                string_quote = ch
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    continue
                opener = stack.pop()
                if not _is_matching_pair(opener, ch):
                    # Keep scanning to avoid false positives
                    stack.clear()
                    continue
                if not stack:
                    # Complete JSON region
                    return text[start: i + 1]  # fmt: skip

    return None


def _is_matching_pair(opening: str, closing: str) -> bool:
    return (opening == "[" and closing == "]") or (opening == "{" and closing == "}")


def _estimate_max_nesting_depth(text: str) -> int:
    """Estimate maximum nesting depth for arrays/objects, ignoring string contents."""
    depth = 0
    max_depth = 0
    string_quote: str | None = None
    escaping = False

    for ch in text:
        if string_quote is not None:
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == string_quote:
                string_quote = None
            continue

        if ch in ('"', "'"):
            string_quote = ch
            continue

        if ch in "[{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
            continue
        if ch in "]}":
            if depth > 0:
                depth -= 1

    return max_depth


def _cleanup_json_string(text: str) -> str:
    """Apply minimal, structure-preserving cleanup for typical LLM JSON quirks."""
    without_comments = _strip_inline_comments(text)
    no_trailing_commas = _remove_trailing_commas(without_comments)
    normalized_quotes = _convert_single_quotes_to_double(no_trailing_commas)
    return normalized_quotes


def _strip_inline_comments(text: str) -> str:
    """Remove // and # comments that are outside of string literals."""
    result: list[str] = []
    i = 0
    n = len(text)
    string_quote: str | None = None
    escaping = False

    while i < n:
        ch = text[i]

        if string_quote is not None:
            result.append(ch)
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == string_quote:
                string_quote = None
            i += 1
            continue

        if ch in ('"', "'"):
            string_quote = ch
            result.append(ch)
            i += 1
            continue

        # Handle // comments
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        # Handle # comments (treat as line comments when outside strings)
        if ch == "#":
            i += 1
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before '}' or ']' outside strings."""
    out: list[str] = []
    string_quote: str | None = None
    escaping = False

    for ch in text:
        if string_quote is not None:
            out.append(ch)
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == string_quote:
                string_quote = None
            continue

        if ch in ('"', "'"):
            string_quote = ch
            out.append(ch)
            continue

        if ch in "]}":
            # Remove any comma immediately before this closing bracket (ignoring whitespace)
            j = len(out) - 1
            while j >= 0 and out[j].isspace():
                j -= 1
            if j >= 0 and out[j] == ",":
                # Drop everything after the comma (spaces) and the comma itself
                k = j
                while k < len(out) and out[k].isspace():
                    k += 1
                del out[j:]
            out.append(ch)
            continue

        out.append(ch)

    return "".join(out)


def _convert_single_quotes_to_double(text: str) -> str:
    """Convert single-quoted strings/keys to double-quoted form when safe.

    This is a best-effort normalization to handle LLM outputs like:
        { 'a': 1, 'b': 'x' }
    It avoids altering content inside already double-quoted strings.
    """
    out: list[str] = []
    in_double = False
    in_single = False
    escaping = False

    for ch in text:
        if in_double:
            out.append(ch)
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == '"':
                in_double = False
            continue

        if in_single:
            if ch == "\\":
                # Preserve escape sequences in single-quoted mode
                out.append(ch)
                escaping = not escaping
                continue
            if ch == '"' and not escaping:
                # Escape bare double-quotes inside the would-be double-quoted string
                out.append('\\"')
                continue
            if ch == "'" and not escaping:
                out.append('"')
                in_single = False
                continue
            out.append(ch)
            escaping = False
            continue

        if ch == '"':
            in_double = True
            out.append(ch)
            continue
        if ch == "'":
            in_single = True
            out.append('"')
            continue

        out.append(ch)

    return "".join(out)


__all__ = [
    "DEFAULT_MAX_SIZE_BYTES",
    "DEFAULT_MAX_DEPTH",
    "loads",
    "loads_from_llm_output",
    "extract_json_from_text",
]


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
        # Unknown string forms are ignored for simplicity
        return None
    return expected
