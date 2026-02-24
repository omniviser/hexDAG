#!/usr/bin/env python3
"""Flag active usage of deprecated APIs in hexdag/.

Scans for ``.. deprecated::`` RST markers in module and class docstrings, then
searches the codebase for imports or references to those deprecated items.

Deprecated items that still have active consumers will never be removed, creating
permanent technical debt.  This check applies migration pressure.

Configuration is loaded from ``.check_deprecated_usage.yaml`` (allowlist).

Usage:
    uv run python scripts/check_deprecated_usage.py
    uv run python scripts/check_deprecated_usage.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ============================================================================
# Config
# ============================================================================

_CONFIG_FILE = ".check_deprecated_usage.yaml"

_DEFAULT_SCAN_DIR = "hexdag"


def _load_config() -> dict[str, Any]:
    """Load allowlist configuration.

    Returns
    -------
    dict
        Configuration with ``allowed_usages`` key.
    """
    config_path = Path(_CONFIG_FILE)
    if not config_path.exists():
        return {"allowed_usages": {}}

    if yaml is None:
        print(f"Warning: PyYAML not installed; ignoring {_CONFIG_FILE}")
        return {"allowed_usages": {}}

    with Path.open(config_path) as f:
        return yaml.safe_load(f) or {}


# ============================================================================
# Models
# ============================================================================


@dataclass
class DeprecatedItem:
    """A deprecated class, function, or module."""

    name: str
    kind: str  # "module", "class", "function"
    file_path: str
    line: int
    replacement: str


@dataclass
class DeprecatedUsage:
    """A usage of a deprecated item."""

    item: DeprecatedItem
    usage_file: str
    usage_line: int
    usage_text: str


# ============================================================================
# Deprecated item discovery
# ============================================================================


_DEPRECATED_RE = re.compile(
    r"\.\.\s+deprecated::\s*\n((?:[ \t]+\S[^\n]*\n)*)",
    re.MULTILINE,
)

# Match RST references like :class:`~hexdag.foo.Bar` or ``Bar``
_RST_REF_RE = re.compile(r":class:`~?([\w.]+)`|``([\w.]+)``")

# Match "Use X instead" or "Prefer X" patterns
_USE_INSTEAD_RE = re.compile(r"(?:Use|Prefer|Import from)\s+(\S+)")


def _extract_replacement(deprecated_block: str) -> str:
    """Extract the suggested replacement from a ``.. deprecated::`` block.

    Parameters
    ----------
    deprecated_block
        The indented text following ``.. deprecated::``.

    Returns
    -------
    str
        A human-readable replacement suggestion, or "(see docstring)".
    """
    # Try to find :class:`...` or ``...`` reference
    match = _RST_REF_RE.search(deprecated_block)
    if match:
        return match.group(1) or match.group(2) or "(see docstring)"

    # Try "Use X instead" / "Prefer X" pattern
    match = _USE_INSTEAD_RE.search(deprecated_block)
    if match:
        return match.group(1).rstrip(".")

    # Fall back to first non-empty line
    for line in deprecated_block.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]

    return "(see docstring)"


def _find_deprecated_modules(scan_dir: Path, root: Path) -> list[DeprecatedItem]:
    """Find modules whose module-level docstring contains ``.. deprecated::``."""
    items: list[DeprecatedItem] = []

    for py_file in sorted(scan_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Check module docstring (first non-empty content)
        rel = str(py_file.relative_to(root))

        # Module-level deprecated marker
        match = _DEPRECATED_RE.search(source[:2000])  # First 2000 chars
        if match:
            replacement = _extract_replacement(match.group(1))
            # Module name is the file's dotted path
            module_name = (
                str(py_file.relative_to(root))
                .replace("/", ".")
                .replace(".py", "")
                .replace(".__init__", "")
            )
            items.append(
                DeprecatedItem(
                    name=module_name,
                    kind="module",
                    file_path=rel,
                    line=source[: match.start()].count("\n") + 1,
                    replacement=replacement,
                )
            )

        # Class-level deprecated markers
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                continue
            dep_match = _DEPRECATED_RE.search(docstring)
            if dep_match:
                replacement = _extract_replacement(dep_match.group(1))
                items.append(
                    DeprecatedItem(
                        name=node.name,
                        kind="class",
                        file_path=rel,
                        line=node.lineno,
                        replacement=replacement,
                    )
                )

    return items


# ============================================================================
# Usage detection
# ============================================================================


def _find_usages(
    item: DeprecatedItem,
    scan_dir: Path,
    root: Path,
    allowed: set[str],
) -> list[DeprecatedUsage]:
    """Find all imports/references to a deprecated item.

    Parameters
    ----------
    item
        The deprecated item to search for.
    scan_dir
        Directory to scan.
    root
        Repository root.
    allowed
        Set of file paths (relative) that are allowed to reference this item.

    Returns
    -------
    list[DeprecatedUsage]
        List of usages found.
    """
    usages: list[DeprecatedUsage] = []

    # Build search patterns
    if item.kind == "module":
        # For deprecated modules, look for imports from that module
        # e.g., "from hexdag.kernel.pipeline_builder import"
        # or "import hexdag.kernel.pipeline_builder"
        patterns = [
            re.compile(rf"from\s+{re.escape(item.name)}\s+import"),
            re.compile(rf"import\s+{re.escape(item.name)}\b"),
        ]
    else:
        # For deprecated classes, look for the class name in imports
        patterns = [
            re.compile(rf"import\s+.*\b{re.escape(item.name)}\b"),
            re.compile(rf"from\s+\S+\s+import\s+.*\b{re.escape(item.name)}\b"),
            # Direct usage as base class
            re.compile(rf"class\s+\w+\({re.escape(item.name)}\)"),
            re.compile(rf"class\s+\w+\(.*\b{re.escape(item.name)}\b.*\)"),
        ]

    for py_file in sorted(scan_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        rel = str(py_file.relative_to(root))

        # Skip the definition file itself
        if rel == item.file_path:
            continue

        # Skip allowed files
        if rel in allowed:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Track TYPE_CHECKING blocks to optionally skip them
        in_type_checking = False
        type_checking_indent = 0

        for line_num, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()

            # Track TYPE_CHECKING blocks
            if re.match(r"if\s+TYPE_CHECKING\s*:", stripped):
                in_type_checking = True
                type_checking_indent = len(line) - len(line.lstrip())
                continue

            if in_type_checking:
                current_indent = len(line) - len(line.lstrip())
                if stripped and current_indent <= type_checking_indent:
                    in_type_checking = False

            # Skip TYPE_CHECKING-guarded imports (type-only usage is OK)
            if in_type_checking:
                continue

            # Skip comments
            if stripped.startswith("#"):
                continue

            for pattern in patterns:
                if pattern.search(stripped):
                    usages.append(
                        DeprecatedUsage(
                            item=item,
                            usage_file=rel,
                            usage_line=line_num,
                            usage_text=stripped,
                        )
                    )
                    break  # One match per line is enough

    return usages


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Check for active usage of deprecated APIs.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Flag usage of deprecated APIs in hexdag/")
    parser.add_argument(
        "--path",
        default=_DEFAULT_SCAN_DIR,
        help="Directory to scan (default: hexdag/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    scan_dir = Path(args.path).resolve()
    if not scan_dir.exists():
        print(f"Error: Path does not exist: {scan_dir}")
        return 1

    root = Path.cwd().resolve()
    config = _load_config()
    allowed_usages: dict[str, list[str]] = config.get("allowed_usages", {}) or {}

    # Step 1: Find all deprecated items
    print("🔍 Scanning for deprecated APIs...")
    deprecated_items = _find_deprecated_modules(scan_dir, root)

    if args.verbose:
        print(f"   Found {len(deprecated_items)} deprecated item(s):")
        for item in deprecated_items:
            print(f"   - {item.kind} '{item.name}' in {item.file_path}:{item.line}")
            print(f"     Replacement: {item.replacement}")

    if not deprecated_items:
        print("✅ No deprecated APIs found")
        return 0

    # Step 2: Find usages of each deprecated item
    all_usages: list[DeprecatedUsage] = []
    for item in deprecated_items:
        allowed_files = set(allowed_usages.get(item.name, []) or [])
        usages = _find_usages(item, scan_dir, root, allowed_files)
        all_usages.extend(usages)

        if args.verbose and not usages:
            print(f"   ✓ No active usages of '{item.name}'")

    # Step 3: Report
    if not all_usages:
        n = len(deprecated_items)
        print(f"✅ No active usage of deprecated APIs ({n} deprecated items, 0 usages)")
        return 0

    # Group by deprecated item
    by_item: dict[str, list[DeprecatedUsage]] = {}
    for usage in all_usages:
        key = usage.item.name
        by_item.setdefault(key, []).append(usage)

    print(f"\n❌ Found {len(all_usages)} usage(s) of deprecated APIs:")
    print("=" * 80)

    for _item_name, usages in sorted(by_item.items()):
        item = usages[0].item
        print(f"\n  {item.kind} '{item.name}' (defined in {item.file_path})")
        print(f"  Replacement: {item.replacement}")
        print(f"  Usages ({len(usages)}):")
        for usage in usages:
            print(f"    {usage.usage_file}:{usage.usage_line}: {usage.usage_text}")

    print("\n" + "=" * 80)
    print("💡 To fix:")
    print("   1. Migrate to the suggested replacement")
    print("   2. Or add to .check_deprecated_usage.yaml allowlist")

    return 1


if __name__ == "__main__":
    sys.exit(main())
