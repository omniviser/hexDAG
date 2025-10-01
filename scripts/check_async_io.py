#!/usr/bin/env python3
"""Static analysis tool to detect synchronous I/O operations in async code.

This script scans Python files for common synchronous I/O patterns that could
block the async event loop. It's designed to run as a pre-commit hook or in CI/CD.

Async-First Architecture Pillar Enforcement:
- Detects blocking file I/O (open, read, write)
- Detects blocking network calls (requests, urllib)
- Detects blocking database operations (sqlite3.connect, cursor.execute)
- Detects synchronous sleep calls
- Ensures async functions use async I/O patterns
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncIOViolation:
    """Represents a detected synchronous I/O operation."""

    file: str
    line: int
    column: int
    operation: str
    message: str
    severity: str = "error"  # "error" or "warning"


class AsyncIOChecker(ast.NodeVisitor):
    """AST visitor to detect synchronous I/O in async contexts."""

    # Patterns that indicate blocking I/O operations
    BLOCKING_CALLS = {
        # File I/O
        "open": "Use aiofiles.open() instead of open()",
        "read": "Use async file.read() instead of sync read()",
        "write": "Use async file.write() instead of sync write()",
        "readlines": "Use async file iteration instead of readlines()",
        # Network I/O
        "requests.get": "Use aiohttp.ClientSession.get() instead of requests.get()",
        "requests.post": "Use aiohttp.ClientSession.post() instead of requests.post()",
        "requests.put": "Use aiohttp.ClientSession.put() instead of requests.put()",
        "requests.delete": "Use aiohttp.ClientSession.delete() instead of requests.delete()",
        "urllib.request.urlopen": "Use aiohttp instead of urllib.request.urlopen()",
        # Database I/O
        "sqlite3.connect": "Use aiosqlite.connect() instead of sqlite3.connect()",
        "cursor.execute": "Use await cursor.execute() with aiosqlite",
        "connection.commit": "Use await connection.commit() with async DB",
        # Sleep - only non-asyncio versions
        "time.sleep": "Use await asyncio.sleep() instead of time.sleep()",
    }

    # Calls that look blocking but are actually async-safe
    SAFE_ASYNC_CALLS = {
        "asyncio.sleep",  # Already async
        "asyncio.wait",
        "asyncio.gather",
        "asyncio.create_task",
    }

    # Allowed sync operations in specific contexts
    ALLOWED_SYNC_PATTERNS = {
        "__init__": ["open", "sqlite3.connect"],  # Init can be sync for local adapters
        "get_available_tools": ["open"],  # Sync version of methods
        "get_tool_schema": ["open"],
        "get_all_tool_schemas": ["open"],
    }

    def __init__(self, filename: str) -> None:
        """Initialize the checker.

        Args
        ----
            filename: Path to the file being checked
        """
        self.filename = filename
        self.violations: list[SyncIOViolation] = []
        self.in_async_function = False
        self.current_function: str | None = None
        self.async_function_stack: list[str] = []
        self.node_stack: list[ast.AST] = []

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        """Visit async function definitions."""
        self.async_function_stack.append(node.name)
        self.in_async_function = True
        self.current_function = node.name

        # Check function body for sync I/O
        self.generic_visit(node)

        self.async_function_stack.pop()
        self.in_async_function = len(self.async_function_stack) > 0
        self.current_function = self.async_function_stack[-1] if self.async_function_stack else None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Visit regular function definitions."""
        prev_function = self.current_function
        self.current_function = node.name

        # Check if we're still in an async context (nested function)
        self.generic_visit(node)

        self.current_function = prev_function

    def _is_allowed_in_context(self, operation: str) -> bool:
        """Check if operation is allowed in current context.

        Args
        ----
            operation: The operation name being checked

        Returns
        -------
            True if operation is allowed in this context
        """
        if not self.current_function:
            return False

        # Check if current function allows this operation
        for pattern, allowed_ops in self.ALLOWED_SYNC_PATTERNS.items():
            if pattern in self.current_function and any(
                allowed in operation for allowed in allowed_ops
            ):
                return True

        return False

    def _is_awaited(self, node: ast.Call) -> bool:
        """Check if a call is being awaited.

        Args
        ----
            node: AST Call node

        Returns
        -------
            True if the call is inside an await expression
        """
        # Check if any parent in the node stack is an Await node
        for parent in reversed(self.node_stack):
            # Check if this call is the value being awaited
            if isinstance(parent, ast.Await) and hasattr(parent, "value") and parent.value == node:
                return True
        return False

    def visit(self, node: ast.AST) -> None:
        """Override visit to track node stack for await detection."""
        self.node_stack.append(node)
        try:
            super().visit(node)
        finally:
            self.node_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Visit function calls to detect blocking operations."""
        # Get the full call name (e.g., "requests.get", "open")
        call_name = self._get_call_name(node)

        if call_name:
            # Skip safe async calls
            if any(call_name.startswith(safe) for safe in self.SAFE_ASYNC_CALLS):
                self.generic_visit(node)
                return

            # Skip if this call is being awaited (it's async)
            if self._is_awaited(node):
                self.generic_visit(node)
                return

            # Check if this is a blocking operation
            for blocking_pattern, suggestion in self.BLOCKING_CALLS.items():
                if (
                    (call_name == blocking_pattern or call_name.endswith(f".{blocking_pattern}"))
                    and self.in_async_function
                    and not self._is_allowed_in_context(call_name)
                ):
                    severity = "error"
                    message = (
                        f"Blocking I/O in async function "
                        f"'{self.current_function}': {call_name}(). {suggestion}"
                    )

                    self.violations.append(
                        SyncIOViolation(
                            file=self.filename,
                            line=node.lineno,
                            column=node.col_offset,
                            operation=call_name,
                            message=message,
                            severity=severity,
                        )
                    )

        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the full name of a function call.

        Args
        ----
            node: AST Call node

        Returns
        -------
            Full function name or None
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            # Handle chained attributes like requests.get
            parts: list[str] = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            return ".".join(parts) if parts else None
        return None


def check_file(file_path: Path, verbose: bool = False) -> list[SyncIOViolation]:
    """Check a single Python file for sync I/O violations.

    Args
    ----
        file_path: Path to Python file
        verbose: Print detailed information

    Returns
    -------
        List of violations found
    """
    try:
        with Path.open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        checker = AsyncIOChecker(str(file_path))
        checker.visit(tree)

        if verbose and checker.violations:
            print(f"\n{file_path}:")
            for violation in checker.violations:
                print(f"  Line {violation.line}: {violation.message}")

        return checker.violations

    except SyntaxError as e:
        if verbose:
            print(f"Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        if verbose:
            print(f"Error processing {file_path}: {e}")
        return []


def check_directory(
    directory: Path, exclude_patterns: list[str] | None = None, verbose: bool = False
) -> list[SyncIOViolation]:
    """Check all Python files in directory for sync I/O violations.

    Args
    ----
        directory: Root directory to scan
        exclude_patterns: List of path patterns to exclude
        verbose: Print detailed information

    Returns
    -------
        List of all violations found
    """
    exclude_patterns = exclude_patterns or []
    all_violations = []

    for py_file in directory.rglob("*.py"):
        # Skip excluded paths
        skip = False
        for pattern in exclude_patterns:
            if re.search(pattern, str(py_file)):
                skip = True
                break

        if skip:
            continue

        violations = check_file(py_file, verbose=verbose)
        all_violations.extend(violations)

    return all_violations


def main() -> int:
    """Main entry point for the async I/O checker.

    Returns
    -------
        Exit code (0 for success, 1 for violations found)
    """
    parser = argparse.ArgumentParser(
        description="Check for synchronous I/O operations in async code"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["hexai"],
        help="Paths to check (files or directories)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude pattern (can be repeated)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--error-only",
        action="store_true",
        help="Only report errors, not warnings",
    )

    args = parser.parse_args()

    # Default exclusions
    default_excludes = [
        r"tests/",
        r"examples/",
        r"hexai_plugins/",
        r"__pycache__/",
        r"\.venv/",
        r"build/",
        r"dist/",
    ]
    exclude_patterns = default_excludes + args.exclude

    all_violations = []

    for path_str in args.paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            violations = check_file(path, verbose=args.verbose)
            all_violations.extend(violations)
        elif path.is_dir():
            violations = check_directory(path, exclude_patterns, verbose=args.verbose)
            all_violations.extend(violations)

    # Filter by severity if requested
    if args.error_only:
        all_violations = [v for v in all_violations if v.severity == "error"]

    # Report results
    if all_violations:
        print("\n" + "=" * 80)
        print("ASYNC I/O VIOLATIONS DETECTED")
        print("=" * 80)

        # Group by file
        violations_by_file: dict[str, list[SyncIOViolation]] = {}
        for v in all_violations:
            violations_by_file.setdefault(v.file, []).append(v)

        for file_path, violations in sorted(violations_by_file.items()):
            print(f"\n{file_path}:")
            for v in sorted(violations, key=lambda x: x.line):
                severity_marker = "ERROR" if v.severity == "error" else "WARNING"
                print(f"  {severity_marker} Line {v.line}:{v.column} - {v.message}")

        print("\n" + "=" * 80)
        print(f"Total violations: {len(all_violations)}")
        print("=" * 80)
        return 1

    if args.verbose:
        print("✓ No async I/O violations found!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
