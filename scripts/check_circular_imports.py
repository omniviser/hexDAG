#!/usr/bin/env python3
"""Detect circular import chains within hexdag/.

This script builds a directed import graph from all Python files and detects
cycles using DFS.  It also reports ``TYPE_CHECKING`` workarounds (where a module
imports under ``if TYPE_CHECKING:`` and falls back to ``Any``) as hidden
cycles — they indicate latent circular dependencies that are masked, not resolved.

Configuration is loaded from .check_circular_imports.yaml (allowlist).

Usage:
    uv run python scripts/check_circular_imports.py
    uv run python scripts/check_circular_imports.py --path hexdag/kernel
    uv run python scripts/check_circular_imports.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ============================================================================
# Config
# ============================================================================

_CONFIG_FILE = ".check_circular_imports.yaml"


def _load_config() -> dict[str, Any]:
    """Load allowlist configuration.

    Returns
    -------
    dict
        Configuration dictionary with ``allowed_cycles`` and
        ``allowed_hidden_cycles`` keys.
    """
    config_path = Path(_CONFIG_FILE)
    if not config_path.exists():
        return {"allowed_cycles": [], "allowed_hidden_cycles": []}

    if yaml is None:
        print(f"Warning: PyYAML not installed; ignoring {_CONFIG_FILE}")
        return {"allowed_cycles": [], "allowed_hidden_cycles": []}

    with Path.open(config_path) as f:
        return yaml.safe_load(f) or {}


# ============================================================================
# Import graph construction
# ============================================================================


def _file_to_module(file_path: Path, root: Path) -> str:
    """Convert a file path to a dotted module name.

    Parameters
    ----------
    file_path
        Absolute or relative ``.py`` file path.
    root
        Repository root (parent of ``hexdag/``).

    Returns
    -------
    str
        Dotted module name, e.g. ``hexdag.kernel.domain.dag``.
    """
    relative = file_path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_inside_type_checking(node: ast.AST, type_checking_ranges: list[tuple[int, int]]) -> bool:
    """Return True if *node* falls within a ``TYPE_CHECKING`` guard."""
    line = getattr(node, "lineno", 0)
    return any(start <= line <= end for start, end in type_checking_ranges)


def _find_type_checking_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Find line ranges of ``if TYPE_CHECKING:`` blocks.

    Returns
    -------
    list[tuple[int, int]]
        List of ``(start_line, end_line)`` for each ``TYPE_CHECKING`` block.
    """
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Match ``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``
        test = node.test
        is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not is_tc:
            continue
        start = node.lineno
        end = max(
            getattr(child, "end_lineno", start)
            for child in ast.walk(node)
            if hasattr(child, "end_lineno")
        )
        ranges.append((start, end))
    return ranges


def _extract_imports(
    file_path: Path,
    root: Path,
    package_prefix: str,
) -> tuple[set[str], list[tuple[str, str, int]]]:
    """Extract runtime imports and TYPE_CHECKING-guarded imports from a file.

    Parameters
    ----------
    file_path
        Python source file.
    root
        Repository root.
    package_prefix
        Only track imports starting with this prefix (e.g. ``hexdag``).

    Returns
    -------
    tuple[set[str], list[tuple[str, str, int]]]
        ``(runtime_imports, hidden_imports)`` where ``runtime_imports`` is a set
        of dotted module names and ``hidden_imports`` is a list of
        ``(module, imported_name, line_number)`` tuples for imports inside
        ``TYPE_CHECKING`` blocks.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return set(), []

    tc_ranges = _find_type_checking_ranges(tree)
    runtime_imports: set[str] = set()
    hidden_imports: list[tuple[str, str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(package_prefix):
                    continue
                if _is_inside_type_checking(node, tc_ranges):
                    hidden_imports.append((alias.name, alias.name, node.lineno))
                else:
                    runtime_imports.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if not node.module or not node.module.startswith(package_prefix):
                continue
            if _is_inside_type_checking(node, tc_ranges):
                hidden_imports.extend(
                    (node.module, alias.name, node.lineno) for alias in (node.names or [])
                )
            else:
                runtime_imports.add(node.module)

    return runtime_imports, hidden_imports


def _resolve_module(imported: str, known_modules: set[str]) -> str | None:
    """Resolve an imported dotted name to the longest matching known module.

    For ``from hexdag.kernel.ports.llm import LLM`` the imported module is
    ``hexdag.kernel.ports.llm``.  If that isn't a known module (perhaps it's
    a package), we try progressively shorter prefixes.

    Returns
    -------
    str or None
        The resolved module name, or None if not found.
    """
    parts = imported.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in known_modules:
            return candidate
    return None


def build_import_graph(
    check_path: Path,
    root: Path,
    package_prefix: str = "hexdag",
) -> tuple[
    dict[str, set[str]],
    dict[str, list[tuple[str, str, int]]],
    set[str],
]:
    """Build the import graph for all Python files under *check_path*.

    Returns
    -------
    tuple
        ``(graph, hidden_imports_by_module, known_modules)``
    """
    graph: dict[str, set[str]] = defaultdict(set)
    hidden_by_module: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    known_modules: set[str] = set()

    # First pass — collect all known modules
    for py_file in check_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        mod = _file_to_module(py_file, root)
        known_modules.add(mod)

    # Second pass — build edges
    for py_file in check_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        mod = _file_to_module(py_file, root)
        runtime_imports, hidden_imports = _extract_imports(py_file, root, package_prefix)

        for imp in runtime_imports:
            resolved = _resolve_module(imp, known_modules)
            if resolved and resolved != mod:
                graph[mod].add(resolved)

        for imp_module, imp_name, lineno in hidden_imports:
            resolved = _resolve_module(imp_module, known_modules)
            if resolved and resolved != mod:
                hidden_by_module[mod].append((resolved, imp_name, lineno))

    return dict(graph), dict(hidden_by_module), known_modules


# ============================================================================
# Cycle detection
# ============================================================================


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find all unique elementary cycles using DFS.

    Returns
    -------
    list[list[str]]
        List of cycles, each as a list of module names forming the cycle.
    """
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []
    cycles: list[list[str]] = []
    seen_cycles: set[frozenset[str]] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                idx = path.index(neighbor)
                cycle = path[idx:]
                cycle_key = frozenset(cycle)
                if cycle_key not in seen_cycles:
                    seen_cycles.add(cycle_key)
                    cycles.append([*cycle, neighbor])

        path.pop()
        rec_stack.discard(node)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)

    return cycles


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Normalize a cycle for comparison (rotate to smallest element first).

    Parameters
    ----------
    cycle
        Cycle path including the repeated start node at the end.

    Returns
    -------
    tuple[str, ...]
        Canonicalized cycle (without the trailing duplicate).
    """
    c = cycle[:-1]  # Drop trailing duplicate
    if not c:
        return ()
    min_idx = c.index(min(c))
    rotated = c[min_idx:] + c[:min_idx]
    return tuple(rotated)


def _has_path(graph: dict[str, set[str]], source: str, target: str) -> bool:
    """Return True if there is a directed path from *source* to *target*.

    Uses BFS to avoid stack overflow on deep graphs.
    """
    visited: set[str] = set()
    queue = [source]
    while queue:
        node = queue.pop(0)
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        queue.extend(graph.get(node, set()) - visited)
    return False


def _is_allowed_cycle(cycle: list[str], allowed: list[list[str]]) -> bool:
    """Check if a cycle matches an allowlisted cycle."""
    norm = _normalize_cycle(cycle)
    for allowed_cycle in allowed:
        if _normalize_cycle(allowed_cycle + [allowed_cycle[0]]) == norm:
            return True
    return False


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Detect circular imports and hidden TYPE_CHECKING workarounds.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Detect circular import chains within hexdag/")
    parser.add_argument(
        "--path",
        default="hexdag",
        help="Root directory to scan (default: hexdag/)",
    )
    parser.add_argument(
        "--no-hidden",
        action="store_true",
        help="Skip reporting hidden (TYPE_CHECKING) cycles",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    check_path = Path(args.path).resolve()
    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}")
        return 1

    root = Path.cwd().resolve()
    config = _load_config()
    allowed_cycles: list[list[str]] = config.get("allowed_cycles", []) or []
    allowed_hidden: list[str] = config.get("allowed_hidden_cycles", []) or []
    allowed_hidden_set = set(allowed_hidden)

    print("🔍 Building import graph...")
    graph, hidden_by_module, known_modules = build_import_graph(check_path, root)

    if args.verbose:
        print(f"   Scanned {len(known_modules)} modules")
        edge_count = sum(len(v) for v in graph.values())
        print(f"   Found {edge_count} import edges")

    # Detect real cycles
    cycles = _find_cycles(graph)
    real_violations = [c for c in cycles if not _is_allowed_cycle(c, allowed_cycles)]

    # Detect hidden cycles (TYPE_CHECKING workarounds)
    # Only report TYPE_CHECKING imports that WOULD create a cycle if moved
    # to runtime.  This filters out the common case of guarding heavy imports
    # for type annotations (which are fine and not hiding real cycles).
    hidden_warnings: list[str] = []
    if not args.no_hidden:
        for mod, imports in sorted(hidden_by_module.items()):
            for target_mod, name, lineno in imports:
                if mod in allowed_hidden_set:
                    continue
                resolved = _resolve_module(target_mod, known_modules)
                if not resolved or resolved == mod:
                    continue
                # Check: would adding this edge create a cycle?
                # i.e., is there already a path from resolved → mod in the graph?
                if _has_path(graph, resolved, mod):
                    file_path = mod.replace(".", "/") + ".py"
                    hidden_warnings.append(
                        f"  {file_path}:{lineno}: imports {name} from {target_mod} "
                        f"under TYPE_CHECKING guard (would create cycle: "
                        f"{resolved} → ... → {mod} → {resolved})"
                    )

    # Report results
    if real_violations:
        print(f"\n❌ Found {len(real_violations)} circular import chain(s):")
        print("=" * 80)
        for i, cycle in enumerate(real_violations, 1):
            chain = " → ".join(cycle)
            print(f"\n  {i}. {chain}")
        print("\n" + "=" * 80)
        print("💡 To fix:")
        print("   1. Move shared types to a dedicated module (e.g., types.py)")
        print("   2. Use dependency injection instead of direct imports")
        print("   3. Add to .check_circular_imports.yaml if intentional")

    if hidden_warnings:
        print(f"\n⚠️  Found {len(hidden_warnings)} hidden cycle(s) (TYPE_CHECKING workarounds):")
        print("-" * 80)
        for warning in hidden_warnings:
            print(warning)
        print("-" * 80)
        print("💡 Hidden cycles use `if TYPE_CHECKING: import X else: X = Any`")
        print("   This masks circular dependencies — isinstance() checks won't work.")

    if not real_violations and not hidden_warnings:
        print("✅ No circular imports detected")
        return 0

    if not real_violations and hidden_warnings:
        print(f"\n✅ No real circular imports (but {len(hidden_warnings)} hidden cycle warnings)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
