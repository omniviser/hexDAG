"""Secure prompt sanitization for hexAI.

Features:

Unicode normalization and removal of control/BIDI marks
Escaping template-breaking chars: { } < > [ ]
Length validation with graceful truncation
API:

SanitizationConfig: pipeline settings
sanitize_text(text, cfg): sanitize single string
parse_sanitization_config(raw): parse YAML 'sanitization' section
sanitize_mapping(value, cfg): recursively sanitize nested dict/list/tuple (strings only)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

CONTROL_CHARACTERS_PATTERN = re.compile(r"[\u0000-\u001F\u007F-\u009F]")

FORMAT_CHARACTERS = [
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u061c",  # ALM
]
FORMAT_CHARACTERS_PATTERN = re.compile("[" + "".join(FORMAT_CHARACTERS) + "]")

ESCAPE_CHARACTERS = {
    "{": r"\{",
    "}": r"\}",
    "<": r"\<",
    ">": r"\>",
    "[": r"\[",
    "]": r"\]",
}

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


@dataclass
class SanitizationConfig:
    """
    Configuration for the prompt sanitization pipeline.

    Attributes
    ---------
    use_sanitizer: bool
        Enables or disables sanitization entirely
    max_input_length: Optional[int]
        Maximum allowed input. None or 0 means no limit. If exceeded the input will be truncated
    escape_template_chars: bool
        Whether to escape template-breaking characters: { } < > [ ].
    normalize_unicode: bool
        Whether to normalize Unicode text to reduce encoding-based issues.
    normalization_form: str
        Unicode normalization form. Defaults to 'NFKC'
    """

    use_sanitizer: bool = True
    max_input_length: int | None = 1000
    escape_template_chars: bool = True
    normalize_unicode: bool = True
    normalization_form: NormalizationForm = "NFKC"  # Normalization form


def _normalize_unicode(text: str, form: NormalizationForm) -> str:
    """
    Normalize Unicode text.

    Args:
    -----
    text: str
        Text to normalize.
    form: str
        Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').

    Returns:
    --------
    str
        Normalized text.
    """
    return unicodedata.normalize(form, text)


def _remove_control_and_bidi(text: str) -> str:
    """Remove control characters (C0/C1) and bidirectional text (LTR: ->, RTL: <-)."""
    text = CONTROL_CHARACTERS_PATTERN.sub("", text)
    return FORMAT_CHARACTERS_PATTERN.sub("", text)


def _escape_template_chars(text: str) -> str:
    """Escape characters that could break template."""
    if not any(ch in text for ch in ESCAPE_CHARACTERS):
        return text
    return "".join(ESCAPE_CHARACTERS.get(ch, ch) for ch in text)


def _truncate(text: str, max_len: int | None) -> str:
    """Truncate text to a maximum length using an ellipsis suffix when possible."""
    if not max_len or max_len < 0 or len(text) <= max_len:
        return text
    suffix = "…"
    keep = max_len - len(suffix)
    if keep <= 0:
        return text[:max_len]
    return text[:keep] + suffix


def sanitize_text(text: str, cfg: SanitizationConfig) -> str:
    """Apply the full sanitization pipeline to provided text."""
    if not cfg.use_sanitizer:
        return text

    if cfg.normalize_unicode:
        text = _normalize_unicode(text, cfg.normalization_form)

    text = _remove_control_and_bidi(text)

    if cfg.escape_template_chars:
        text = _escape_template_chars(text)

    return _truncate(text, cfg.max_input_length)


# YAML parsing and validation for sanitization

# Accept common boolean-like values.
_TRUTH = {"yes", "true", "on"}
_FALSE = {"no", "false", "off"}


def _coerce_bool(v: Any, field: str) -> bool:
    """Coerce yes/no/true/false/on/off value to boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        val = v.strip().lower()
        if val in _TRUTH:
            return True
        if val in _FALSE:
            return False
        raise ValueError(f"{field} must be yes/no/true/false/on/off, got: {v}")
    raise ValueError(f"{field} must be boolean-like got: {type(v).__name__}")


def _coerce_positive_int_or_none(v: Any, field: str) -> int | None:
    """allow a positive int or null"""
    if v is None:
        return None
    if isinstance(v, int) and v > 0:
        return v
    raise ValueError(f"{field} must be a positive integer or null")


def _coerce_norm_form(v: Any, field: str) -> NormalizationForm:
    """Validate Unicode normalization form and coerce to a strict Literal type."""
    if v is None:
        return "NFKC"
    if not isinstance(v, str):
        raise ValueError(f"{field} must be one of NFC/NFD/NFKC/NFKD, got: {v!r}")
    val = v.strip().upper()
    if val not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"{field} must be one of NFC/NFD/NFKC/NFKD, got: {v!r}")
    return val  # type: ignore[return-value]


def parse_sanitization_config(raw: dict[str, Any] | None) -> SanitizationConfig:
    """
    Parse and validate the 'sanitization' YAML section into SanitizationConfig.

    Supported fields:
      - use_sanitizer: yes/no/true/false/on/off (default True)
      - max_input_length: positive integer or null (default 1000)
      - escape_template_chars: boolean-like (default True)
      - normalize_unicode: boolean-like (default True)
      - normalization_form: NFC/NFD/NFKC/NFKD (default NFKC)

    Raises ValueError with descriptive messages on invalid input.
    """
    if not raw:
        return SanitizationConfig(use_sanitizer=False, max_input_length=1000)

    try:
        use = _coerce_bool(raw.get("use_sanitizer", True), "sanitization.use_sanitizer")
        max_len = _coerce_positive_int_or_none(
            raw.get("max_input_length", 1000), "sanitization.max_input_length"
        )
        esc = _coerce_bool(
            raw.get("escape_template_chars", True), "sanitization.escape_template_chars"
        )
        norm = _coerce_bool(raw.get("normalize_unicode", True), "sanitization.normalize_unicode")
        form = _coerce_norm_form(
            raw.get("normalization_form", "NFKC"), "sanitization.normalization_form"
        )

        return SanitizationConfig(
            use_sanitizer=use,
            max_input_length=max_len,
            escape_template_chars=esc,
            normalize_unicode=norm,
            normalization_form=form,
        )
    except ValueError as e:
        raise ValueError(f"Invalid sanitization config: {e}") from e


# Recursive sanitization of nested inputs
EXCLUDED_KEYS: set[str] = {"_sanitization_cfg", "context_history"}


def sanitize_mapping(value: Any, cfg: SanitizationConfig) -> Any:
    """
    Recursively sanitize all string leaves in nested dicts/lists/tuples,
    skipping values under EXCLUDED_KEYS.

    This is an integration helper. Use it in BaseLLMNode/AgentNode right
    before rendering to ensure every string that reaches the template
    is sanitized consistently.
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k in EXCLUDED_KEYS:
                out[k] = v
            else:
                out[k] = sanitize_mapping(v, cfg)
        return out

    if isinstance(value, (list, tuple)):
        return type(value)(sanitize_mapping(v, cfg) for v in value)

    if isinstance(value, str):
        return sanitize_text(value, cfg)

    return value
