"""Reference resolver — extracts node references from mapping values, expressions, and templates.

Used by the YAML builder and validator to infer dependencies so that
users no longer need to specify redundant ``dependencies``.

Two reference grammars with different scan strategies:

- **Jinja refs** (``{{node.field}}``, ``{{node}}``) — unambiguous
  delimiters, so *every* string in a node spec is scanned at any
  nesting depth (:func:`extract_refs_from_spec`). Custom node fields
  get inference automatically.
- **Bare expression refs** (``node.field`` without delimiters) —
  ambiguous in prose, so only the framework-owned fields listed in
  :data:`EXPRESSION_STRING_FIELDS` / :data:`EXPRESSION_DICT_FIELDS`
  (plus ``input_mapping`` and ``branches[].condition``) are scanned.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from hexdag.kernel.context.execution_context import RESERVED_NAMES
from hexdag.kernel.expression_parser import ALLOWED_FUNCTIONS

if TYPE_CHECKING:
    from collections.abc import Iterator

# Pattern: identifier.identifier (but not $input.field)
_NODE_FIELD_RE = re.compile(r"(?<!\$)\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_.]*")

# Pattern: splits a source string into identifier tokens and non-identifier chars.
# Used to find bare node names in expressions like "analyzer + 1".
_IDENT_SPLIT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Jinja2 variable pattern: {{node.field}} or {{ node.field }}
_JINJA_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_.]*\s*\}\}")

# Jinja2 bare variable pattern: {{node}} — whole-output reference
# (e.g. ``conversation: "{{chat_history}}"``)
_JINJA_BARE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

# First identifier of any {{...}} occurrence (dotted, bare, or call) —
# used by the validator's typo lint, not for dependency inference.
_JINJA_HEAD_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)")

# Spec fields whose *string* value uses bare expression grammar.
EXPRESSION_STRING_FIELDS: frozenset[str] = frozenset({"when", "condition", "items"})

# Spec fields whose *dict* values use bare expression grammar.
EXPRESSION_DICT_FIELDS: frozenset[str] = frozenset({"expressions", "state_update", "payload"})

# Names that are never node references — expression namespaces (from kernel)
# plus Python/YAML keyword-like literals and built-in functions.
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
    for regex in (_JINJA_VAR_RE, _JINJA_BARE_RE):
        for match in regex.finditer(template):
            candidate = match.group(1)
            if candidate in _BUILTIN_NAMES or candidate in RESERVED_NAMES:
                continue
            if candidate in known_nodes:
                refs.add(candidate)
            elif macro_instances:
                for mi in macro_instances:
                    if candidate.startswith(f"{mi}_"):
                        refs.add(mi)
                        break
    return refs


def iter_spec_strings(value: Any) -> Iterator[str]:
    """Yield every string in a node spec, at any nesting depth.

    Walks dicts and lists recursively so templated fields are found
    regardless of where a node factory chose to put them — including
    custom node kinds the framework has never seen.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_spec_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_spec_strings(v)


def extract_refs_from_spec(
    spec: dict[str, Any],
    known_nodes: frozenset[str],
    macro_instances: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract all node references from a node's ``spec``.

    Single source of truth shared by the builder's dependency
    inference and the validator's reference rules, so the two can
    never disagree about what counts as a reference.

    Jinja refs (``{{node.field}}``) are found in *any* string at any
    nesting depth. Bare expression refs (``node.field``) are only
    scanned in the framework-owned expression fields.

    Parameters
    ----------
    spec : dict[str, Any]
        The node's ``spec`` mapping (caller excludes the node's own
        name from *known_nodes* to avoid self-references).
    known_nodes : frozenset[str]
        Node names eligible as reference targets.
    macro_instances : frozenset[str]
        Macro invocation names for prefix-based matching.

    Returns
    -------
    set[str]
        Node names referenced anywhere in the spec.
    """
    refs: set[str] = set()

    # Jinja grammar: every string, anywhere in the spec.
    for text in iter_spec_strings(spec):
        if "{{" in text:
            refs |= extract_refs_from_template(text, known_nodes, macro_instances)

    # Bare expression grammar: framework-owned fields only.
    input_mapping = spec.get("input_mapping")
    if isinstance(input_mapping, dict):
        refs |= extract_refs_from_mapping(input_mapping, known_nodes, macro_instances)

    for key in EXPRESSION_DICT_FIELDS:
        val = spec.get(key)
        if isinstance(val, dict):
            refs |= extract_refs_from_expressions(val, known_nodes, macro_instances)

    for key in EXPRESSION_STRING_FIELDS:
        val = spec.get(key)
        if isinstance(val, str):
            refs |= extract_refs_from_string(val, known_nodes, macro_instances)

    branches = spec.get("branches")
    if isinstance(branches, list):
        for branch in branches:
            if isinstance(branch, dict) and isinstance(branch.get("condition"), str):
                refs |= extract_refs_from_string(branch["condition"], known_nodes, macro_instances)

    return refs


def extract_jinja_head_names(text: str) -> set[str]:
    """Return the first identifier of every ``{{...}}`` occurrence.

    Unlike :func:`extract_refs_from_template` this does *not* filter
    by known node names — it feeds the validator's typo lint, which
    wants the names that did **not** match anything.
    """
    return {m.group(1) for m in _JINJA_HEAD_RE.finditer(text)}


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


# Pattern: $input.field_name — captures the field name after $input.
_INPUT_FIELD_RE = re.compile(r"\$input\.([A-Za-z_][A-Za-z0-9_]*)")


def extract_input_refs_from_mapping(input_mapping: dict[str, Any]) -> set[str]:
    """Extract ``$input.X`` field names from ``input_mapping`` values.

    Unlike :func:`extract_refs_from_mapping` which *skips* ``$input`` references,
    this function *collects* them — returning the field names (without the
    ``$input.`` prefix) so the validator can cross-check them against the
    pipeline's declared ``input_schema`` or sibling nodes.

    Parameters
    ----------
    input_mapping : dict[str, Any]
        Mapping of ``{target_field: "source_path"}``.

    Returns
    -------
    set[str]
        Field names referenced via ``$input.X``.
    """
    fields: set[str] = set()
    for val in input_mapping.values():
        if isinstance(val, str):
            for match in _INPUT_FIELD_RE.finditer(val):
                fields.add(match.group(1))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    for match in _INPUT_FIELD_RE.finditer(item):
                        fields.add(match.group(1))
    return fields


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
        if candidate in RESERVED_NAMES or candidate in _BUILTIN_NAMES:
            continue
        if candidate in known_nodes:
            refs.add(candidate)
        elif sorted_macros:
            for mi in sorted_macros:
                if candidate.startswith(f"{mi}_"):
                    refs.add(mi)
                    break

    # Bare identifiers — detect known nodes and macro-prefixed names that
    # the dotted regex above cannot match.  Handles both simple bare names
    # ("analyzer") and bare names inside expressions ("analyzer + 1").
    # Extract all identifier tokens, then look them up in known_nodes directly.
    # Tokens that are part of a "node.field" chain are skipped because _NODE_FIELD_RE
    # already handled those and they appear with a trailing or leading dot in the
    # source — but we filter by checking the char before/after each token position.
    for token_match in _IDENT_SPLIT_RE.finditer(stripped):
        start = token_match.start()
        end = token_match.end()
        # Skip if preceded by '.' or '$' (part of node.field or $input)
        if start > 0 and stripped[start - 1] in (".", "$"):
            continue
        # Skip if followed by '.' (this is the owner in a node.field pattern)
        if end < len(stripped) and stripped[end] == ".":
            continue
        candidate = token_match.group(0)
        if candidate in _BUILTIN_NAMES or candidate in RESERVED_NAMES:
            continue
        if candidate in known_nodes:
            refs.add(candidate)
        elif sorted_macros:
            for mi in sorted_macros:
                if candidate.startswith(f"{mi}_"):
                    refs.add(mi)
                    break

    return refs
