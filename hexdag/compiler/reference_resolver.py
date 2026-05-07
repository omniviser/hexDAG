"""Reference resolver — extracts node references from mapping values, expressions, and templates.

Used by the YAML builder to infer dependencies from ``input_mapping``,
``expressions``, and ``prompt_template`` / ``template`` fields so that
users no longer need to specify redundant ``dependencies``.
"""

from __future__ import annotations

import re
from typing import Any

from hexdag.kernel.expression_parser import ALLOWED_FUNCTIONS

# Pattern: identifier.identifier (but not $input.field)
_NODE_FIELD_RE = re.compile(r"(?<!\$)\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_.]*")

# Jinja2 variable pattern: {{node.field}} or {{ node.field }}
_JINJA_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_.]*\s*\}\}")

# Reserved prefixes that are never node references
_RESERVED_PREFIXES = frozenset({"$input", "ctx"})

# Names that should never be treated as node references, even if they match
# the ``identifier.identifier`` pattern.  Derived from the expression parser's
# ALLOWED_FUNCTIONS plus Python/YAML keyword-like literals.
_BUILTIN_NAMES = frozenset(ALLOWED_FUNCTIONS.keys()) | frozenset({
    "self",
    "type",
    "true",
    "false",
    "none",
    "True",
    "False",
    "None",
    "null",
    "ctx",
})


def extract_refs_from_mapping(
    input_mapping: dict[str, Any],
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract node names referenced in ``input_mapping`` values.

    Recognises the ``node_name.field`` pattern used by
    :pyclass:`ExecutionCoordinator._apply_input_mapping`.
    Skips ``$input.*`` references (pipeline input, not a node).

    Parameters
    ----------
    input_mapping : dict[str, Any]
        Mapping of ``{target_field: "source_path"}``.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline (used for validation).
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names referenced in the mapping.
    """
    refs: set[str] = set()
    for source_path in input_mapping.values():
        if isinstance(source_path, str):
            refs.update(_extract_node_refs(source_path, known_nodes, macro_instances))
        elif isinstance(source_path, list):
            for item in source_path:
                if isinstance(item, str):
                    refs.update(_extract_node_refs(item, known_nodes, macro_instances))
    return refs


def extract_refs_from_expressions(
    expressions: dict[str, Any],
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract node names referenced in expression strings.

    Scans for ``node_name.field`` tokens where ``node_name`` matches
    a known node.

    Parameters
    ----------
    expressions : dict[str, str]
        Mapping of ``{variable_name: expression_string}``.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline.
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names referenced in the expressions.
    """
    refs: set[str] = set()
    for expr in expressions.values():
        if not isinstance(expr, str):
            continue
        refs.update(_extract_node_refs(expr, known_nodes, macro_instances))
    return refs


def extract_refs_from_template(
    template: str,
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract node names referenced in Jinja2-style templates.

    Looks for ``{{node_name.field}}`` patterns where ``node_name``
    matches a known node.

    Parameters
    ----------
    template : str
        Jinja2 template string.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline.
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names referenced in the template.
    """
    refs: set[str] = set()
    for match in _JINJA_VAR_RE.finditer(template):
        candidate = match.group(1)
        if candidate in _BUILTIN_NAMES:
            continue
        if candidate in known_nodes:
            refs.add(candidate)
        elif macro_instances:
            for mi in macro_instances:
                if candidate.startswith(f"{mi}_"):
                    refs.add(mi)
                    break
    return refs


def extract_refs_from_string(
    source: str,
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract node references from a single string (condition, items, etc.).

    Public wrapper around :func:`_extract_node_refs` for use by
    ``_infer_deps`` when scanning composite node fields.

    Parameters
    ----------
    source : str
        A source path or expression string.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline.
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names found.
    """
    return _extract_node_refs(source, known_nodes, macro_instances)


def _extract_node_refs(
    source: str,
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract node references from a single source string.

    Parameters
    ----------
    source : str
        A source path or expression string.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline.
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names found.
    """
    refs: set[str] = set()

    # Skip pure $input references
    stripped = source.strip()
    if stripped == "$input" or stripped.startswith("$input."):
        return refs

    # Sort macro instances by length descending so longer prefixes match first.
    # Without this, "extract" could greedily match "extract_rate_node" before
    # the correct macro "extract_rate" gets a chance.
    sorted_macros = sorted(macro_instances, key=len, reverse=True) if macro_instances else []

    for match in _NODE_FIELD_RE.finditer(source):
        candidate = match.group(1)
        if candidate in _RESERVED_PREFIXES or candidate in _BUILTIN_NAMES:
            continue
        if candidate in known_nodes:
            refs.add(candidate)
        elif sorted_macros:
            for mi in sorted_macros:
                if candidate.startswith(f"{mi}_"):
                    refs.add(mi)
                    break

    # Bare names (no dot) — detect known nodes and macro-prefixed names
    # that the dotted regex above cannot match.
    if "." not in stripped and stripped.isidentifier() and stripped not in _BUILTIN_NAMES:
        if stripped in known_nodes:
            refs.add(stripped)
        elif sorted_macros:
            for mi in sorted_macros:
                if stripped.startswith(f"{mi}_"):
                    refs.add(mi)
                    break

    return refs
