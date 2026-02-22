"""Semantic sanitized types for declarative output schemas.

Provides a registry of type names usable in ``output_schema`` YAML fields.
Each type wraps a base Python type with a Pydantic ``BeforeValidator`` that
cleans messy LLM output before validation.

Built-in types
--------------
- ``currency``       – strip currency symbols/commas → float
- ``flexible_bool``  – "yes"/"true"/"1" → True
- ``score``          – parse float, clamp to 0.0–1.0
- ``upper_str``      – trim + uppercase
- ``lower_str``      – trim + lowercase
- ``nullable_str``   – "N/A"/"none"/"TBD" etc → None
- ``trimmed_str``    – strip whitespace

Custom registration (Python)::

    from hexdag.kernel.validation.sanitized_types import register_type
    register_type("mc_number", str, lambda v: re.search(r"MC-?\\d+", str(v)).group())

Custom registration (YAML ``custom_types`` block)::

    spec:
      custom_types:
        mc_number:
          base: str
          pattern: "MC-?\\d+"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BeforeValidator

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Null strings
# ---------------------------------------------------------------------------

COMMON_NULLS: frozenset[str] = frozenset({
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "tbd",
    "unknown",
    "not available",
    "not specified",
    "-",
    "--",
    "",
})

# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_BASE_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


@dataclass(frozen=True, slots=True)
class _TypeEntry:
    """Internal entry in the sanitized types registry."""

    annotated: Any
    description: str


_REGISTRY: dict[str, _TypeEntry] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_type(
    name: str,
    base: type,
    cleaner: Callable[[Any], Any],
    description: str = "",
) -> None:
    """Register a sanitized type usable in ``output_schema`` as a string name.

    Parameters
    ----------
    name : str
        Type name (e.g. ``"currency"``). Must not collide with built-in types.
    base : type
        Target Python type (``str``, ``int``, ``float``, ``bool``).
    cleaner : Callable[[Any], Any]
        Function that receives a raw value and returns the cleaned value.
    description : str
        Human-readable description for auto-documentation.
    """
    # Use Union[base, None] so the BeforeValidator can return None (e.g. "N/A" → None)
    # and Pydantic accepts it without a string_type validation error.
    # The union is intentionally dynamic (base is a runtime type parameter).
    from typing import Union

    nullable = Union[base, None]  # type: ignore[valid-type]  # noqa: UP007
    annotated = Annotated[nullable, BeforeValidator(cleaner)]
    _REGISTRY[name] = _TypeEntry(annotated=annotated, description=description)


def register_type_from_config(name: str, config: dict[str, Any]) -> None:
    """Register a sanitized type from a YAML ``custom_types`` config dict.

    Parameters
    ----------
    name : str
        Type name to register.
    config : dict[str, Any]
        Declarative config with keys: ``base`` (required), plus optional
        ``pattern``, ``nulls``, ``strip``, ``clamp``, ``upper``, ``lower``,
        ``trim``, ``max_length``, ``default``, ``true_values``, ``false_values``.
    """
    base_name = config.get("base", "str")
    base = _BASE_TYPE_MAP.get(base_name)
    if base is None:
        valid = ", ".join(sorted(_BASE_TYPE_MAP))
        raise ValueError(
            f"Custom type '{name}': invalid base '{base_name}'. Must be one of: {valid}"
        )

    cleaner = _build_cleaner_from_config(config)
    desc = config.get("description", f"Custom type: {name}")
    register_type(name, base, cleaner, description=desc)


def get_type(name: str) -> Any | None:
    """Look up a registered sanitized type by name.

    Returns
    -------
    Any | None
        ``Annotated[base, BeforeValidator(cleaner)]`` or ``None`` if not found.
    """
    entry = _REGISTRY.get(name)
    return entry.annotated if entry is not None else None


def get_available_types() -> dict[str, str]:
    """Return ``{name: description}`` for all registered sanitized types.

    Used by schema generators, error messages, and MCP documentation.
    """
    return {name: entry.description for name, entry in sorted(_REGISTRY.items())}


# ---------------------------------------------------------------------------
# Config-driven cleaner builder
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _build_cleaner_from_config(config: dict[str, Any]) -> Callable[[Any], Any]:
    """Build a cleaner function from a declarative YAML config dict.

    Processing order (fixed):
    1. null check  2. trim  3. regex extract  4. strip chars
    5. type conversion  6. clamp  7. case  8. truncate
    """
    base_name: str = config.get("base", "str")
    base = _BASE_TYPE_MAP.get(base_name, str)

    # Parse config fields
    nulls_cfg = config.get("nulls")
    null_strings: frozenset[str] | None = None
    if nulls_cfg == "common":
        null_strings = COMMON_NULLS
    elif isinstance(nulls_cfg, list):
        null_strings = frozenset(s.lower() for s in nulls_cfg)

    pattern_cfg = config.get("pattern")
    patterns: list[re.Pattern[str]] | None = None
    if isinstance(pattern_cfg, str):
        patterns = [re.compile(pattern_cfg)]
    elif isinstance(pattern_cfg, list):
        patterns = [re.compile(p) for p in pattern_cfg]

    strip_chars: str | None = config.get("strip")
    trim: bool = config.get("trim", True)
    upper: bool = config.get("upper", False)
    lower: bool = config.get("lower", False)
    max_length: int | None = config.get("max_length")
    default = config.get("default", _SENTINEL)

    clamp_cfg = config.get("clamp")
    clamp_min: float | None = None
    clamp_max: float | None = None
    if isinstance(clamp_cfg, list) and len(clamp_cfg) == 2:
        clamp_min, clamp_max = float(clamp_cfg[0]), float(clamp_cfg[1])

    true_values: frozenset[str] | None = None
    false_values: frozenset[str] | None = None
    if config.get("true_values"):
        true_values = frozenset(s.lower() for s in config["true_values"])
    if config.get("false_values"):
        false_values = frozenset(s.lower() for s in config["false_values"])

    def _cleaner(v: Any) -> Any:  # noqa: C901, PLR0911, PLR0912
        # 0. Already None
        if v is None:
            return default if default is not _SENTINEL else None

        # 1. Convert to string for processing
        text = str(v)

        # 2. Trim
        if trim:
            text = text.strip()

        # 3. Null check
        if null_strings is not None and text.lower() in null_strings:
            return default if default is not _SENTINEL else None

        # Empty after trim
        if not text:
            return default if default is not _SENTINEL else None

        # 4. Regex extraction
        if patterns is not None:
            extracted: str | None = None
            for pat in patterns:
                m = pat.search(text)
                if m:
                    extracted = m.group(1) if m.lastindex else m.group(0)
                    break
            if extracted is None:
                return default if default is not _SENTINEL else None
            text = extracted

        # 5. Strip characters
        if strip_chars:
            for ch in strip_chars:
                text = text.replace(ch, "")

        # 6. Type conversion
        result: Any
        if base is bool:
            low = text.lower()
            if true_values and low in true_values:
                result = True
            elif false_values and low in false_values:
                result = False
            elif low in ("true", "yes", "1", "y", "on"):
                result = True
            elif low in ("false", "no", "0", "n", "off"):
                result = False
            else:
                result = text  # let Pydantic handle it
        elif base is float:
            try:
                result = float(text)
            except (ValueError, TypeError):
                return default if default is not _SENTINEL else None
        elif base is int:
            try:
                result = int(float(text))
            except (ValueError, TypeError):
                return default if default is not _SENTINEL else None
        else:
            result = text

        # 7. Clamp (numeric)
        if clamp_min is not None and clamp_max is not None and isinstance(result, (int, float)):
            result = max(clamp_min, min(clamp_max, float(result)))
            if base is int:
                result = int(result)

        # 8. Case conversion (string)
        if isinstance(result, str):
            if upper:
                result = result.upper()
            elif lower:
                result = result.lower()

        # 9. Truncate (string)
        if max_length is not None and isinstance(result, str):
            result = result[:max_length]

        return result

    return _cleaner


# ---------------------------------------------------------------------------
# Built-in cleaners
# ---------------------------------------------------------------------------


def _clean_nullable(v: Any) -> Any:
    """Convert null-like strings to None."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in COMMON_NULLS:
        return None
    return v


def _clean_currency(v: Any) -> Any:
    """Strip currency symbols and commas, convert to float."""
    v = _clean_nullable(v)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    cleaned = re.sub(r"[^\d.\-]", "", str(v))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_flexible_bool(v: Any) -> Any:
    """Accept yes/no/true/false/1/0 as booleans."""
    v = _clean_nullable(v)
    if v is None or isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in ("true", "yes", "1", "y", "on"):
        return True
    if text in ("false", "no", "0", "n", "off"):
        return False
    return v


def _clean_score(v: Any) -> Any:
    """Parse float and clamp to 0.0–1.0."""
    v = _clean_nullable(v)
    if v is None:
        return None
    try:
        val = float(v)
    except (ValueError, TypeError):
        return None
    return max(0.0, min(1.0, val))


def _clean_upper(v: Any) -> Any:
    if v is None:
        return None
    return str(v).strip().upper() or None


def _clean_lower(v: Any) -> Any:
    if v is None:
        return None
    return str(v).strip().lower() or None


def _clean_trimmed(v: Any) -> Any:
    if v is None:
        return None
    return str(v).strip() or None


# ---------------------------------------------------------------------------
# Register built-in types
# ---------------------------------------------------------------------------

register_type(
    "currency",
    float,
    _clean_currency,
    description='Strip currency symbols/commas → float (e.g. "$1,200" → 1200.0)',
)
register_type(
    "flexible_bool",
    bool,
    _clean_flexible_bool,
    description='"yes"/"true"/"1" → True, "no"/"false"/"0" → False',
)
register_type("score", float, _clean_score, description="Parse float and clamp to 0.0–1.0")
register_type("upper_str", str, _clean_upper, description="Trim whitespace + force uppercase")
register_type("lower_str", str, _clean_lower, description="Trim whitespace + force lowercase")
register_type("nullable_str", str, _clean_nullable, description='"N/A", "none", "TBD", etc → None')
register_type("trimmed_str", str, _clean_trimmed, description="Strip leading/trailing whitespace")
