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
_RESERVED_PREFIXES = frozenset({"$input"})

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
})


def extract_refs_from_mapping(
    input_mapping: dict[str, Any],
    known_nodes: frozenset[str],
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

    Returns
    -------
    set[str]
        Node names referenced in the mapping.
    """
    refs: set[str] = set()
    for source_path in input_mapping.values():
        if not isinstance(source_path, str):
            continue
        refs.update(_extract_node_refs(source_path, known_nodes))
    return refs


def extract_refs_from_expressions(
    expressions: dict[str, Any],
    known_nodes: frozenset[str],
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

    Returns
    -------
    set[str]
        Node names referenced in the expressions.
    """
    refs: set[str] = set()
    for expr in expressions.values():
        if not isinstance(expr, str):
            continue
        refs.update(_extract_node_refs(expr, known_nodes))
    return refs


def extract_refs_from_template(
    template: str,
    known_nodes: frozenset[str],
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

    Returns
    -------
    set[str]
        Node names referenced in the template.
    """
    refs: set[str] = set()
    for match in _JINJA_VAR_RE.finditer(template):
        candidate = match.group(1)
        if candidate in known_nodes and candidate not in _BUILTIN_NAMES:
            refs.add(candidate)
    return refs


def _extract_node_refs(source: str, known_nodes: frozenset[str]) -> set[str]:
    """Extract node references from a single source string.

    Parameters
    ----------
    source : str
        A source path or expression string.
    known_nodes : frozenset[str]
        Set of all node names in the pipeline.

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

    for match in _NODE_FIELD_RE.finditer(source):
        candidate = match.group(1)
        if candidate in _RESERVED_PREFIXES or candidate in _BUILTIN_NAMES:
            continue
        if candidate in known_nodes:
            refs.add(candidate)

    return refs
