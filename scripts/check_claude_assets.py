#!/usr/bin/env python3
"""Check that node-kind aliases quoted in bundled Claude Code assets stay in sync with code.

The ``.claude`` scaffold shipped under ``hexdag/cli/templates/claude`` is hand-written prose,
so it is NOT regenerated. But it (and ``CLAUDE.md``) quote the built-in node-kind alias list
(``llm_node``, ``expression_node``, ``transition`` ...). That list is derived from code and
drifts whenever a node factory is added, removed, renamed, or re-aliased.

This check mirrors ``check_schemas.py``: it fails (exit 1) if a doc quotes a node kind that no
longer resolves to a node factory (a stale/removed alias). It does NOT require docs to mention
every kind — omitting a niche alias is fine — it only flags aliases the docs assert that code
no longer backs.

Scope/limitation: it recognises kinds by the ``_node`` suffix (plus the known short alias
``transition``). A doc typo that also breaks the suffix (e.g. ``llm_nodex``) is a plain typo, not
code drift, and is out of scope — matching arbitrary snake_case prose words as "aliases" would
produce false positives. The value here is catching a *renamed/removed real kind* going stale.

Exit codes:
    0: no stale node-kind aliases quoted in the checked docs
    1: a checked doc quotes a node kind that code no longer provides
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The alias registry resolves any builtin alias (nodes, adapters, macros) to a full path.
from hexdag.kernel.resolver import get_builtin_aliases
from hexdag.stdlib.nodes._discovery import discover_node_factories

ROOT = Path(__file__).parent.parent

# Docs that quote the node-kind alias list. Add new asset files here as they appear.
CHECKED_DOCS = [
    ROOT / "hexdag/cli/templates/claude/skills/hexdag-pipeline/SKILL.md",
    ROOT / "hexdag/cli/templates/claude/agents/hexdag-pipeline-debugger.md",
    ROOT / "hexdag/cli/templates/claude/workflows/hexdag-pipeline-review.js",
]

# Aliases that look like node kinds but are intentionally not node factories, or are documented
# for teaching without being a live kind. Keep this empty unless a real exception appears.
KNOWN_NON_NODE_TERMS: frozenset[str] = frozenset()


def live_node_kinds() -> set[str]:
    """Return the set of user-writable node-kind aliases currently backed by code.

    Combines the auto-discovered factory aliases (``discover_node_factories``) with any short
    alias registered in the global alias registry that resolves to a discovered node class
    (e.g. ``transition`` for ``TransitionNode``).
    """
    # Compare by class name (last path segment): discover_node_factories() yields the short
    # re-exported path (``...nodes.TransitionNode``) while get_builtin_aliases() yields the full
    # module path (``...nodes.transition_node.TransitionNode``) for the same class.
    factory_classes = {path.rsplit(".", 1)[-1] for path in discover_node_factories().values()}
    kinds = {alias for alias in discover_node_factories() if ":" not in alias}
    for alias, full_path in get_builtin_aliases().items():
        if ":" in alias:
            continue
        if full_path.rsplit(".", 1)[-1] in factory_classes:
            kinds.add(alias)
    return kinds


def quoted_node_kinds(text: str) -> set[str]:
    """Extract snake_case tokens from a doc that look like node-kind aliases.

    Matches backtick-quoted or bare tokens ending in ``_node`` plus the well-known short
    aliases the docs use. This is deliberately conservative — it only pulls tokens that are
    unambiguously meant as node kinds, so prose words are not misread as aliases.
    """
    node_suffix = set(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)*_node)\b", text))
    # Short aliases the docs may use without the _node suffix (must be word-isolated).
    short = {a for a in ("transition",) if re.search(rf"`{a}`|\b{a}\b", text)}
    return (node_suffix | short) - KNOWN_NON_NODE_TERMS


def main() -> int:
    """Verify no checked doc quotes a node kind that code no longer provides."""
    live = live_node_kinds()
    had_error = False

    for doc in CHECKED_DOCS:
        if not doc.exists():
            print(f"WARN: checked doc missing (skipped): {doc.relative_to(ROOT)}")
            continue
        quoted = quoted_node_kinds(doc.read_text(encoding="utf-8"))
        stale = quoted - live
        if stale:
            had_error = True
            print(f"STALE node-kind alias(es) in {doc.relative_to(ROOT)}:")
            for alias in sorted(stale):
                print(f"  - {alias!r} is documented but no longer resolves to a node factory")

    if had_error:
        print(
            "\nFix: update the doc to use a current node kind, or add the new factory/alias. "
            f"Live node kinds: {', '.join(sorted(live))}"
        )
        return 1

    print(f"check-claude-assets: OK ({len(CHECKED_DOCS)} docs, {len(live)} live node kinds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
