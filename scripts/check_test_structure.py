#!/usr/bin/env python3
"""
Pre-commit hook to validate test structure consistency.

This script ensures that:
1. Every test file corresponds to an existing source file
2. Test files are in the correct location relative to their source files
3. No orphaned test files exist
4. Test files follow the naming convention: test_<module_name>.py

Expected structure:
- Source: hexdag/path/to/module.py
- Test:   tests/hexdag/path/to/test_module.py
"""

import sys
from pathlib import Path
from typing import NamedTuple

# Test files that intentionally don't follow the 1:1 naming convention.
# These test a feature/aspect of an existing module rather than the module itself.
_ALLOWED_ORPHANS: frozenset[str] = frozenset({
    # Tests SupportsKeyValue protocol on InMemoryMemory (source: in_memory_memory.py)
    "tests/hexdag/stdlib/adapters/memory/test_in_memory_key_value.py",
    # Round-trip serialization tests spanning pipeline_run, scheduled_task, entity_state
    "tests/hexdag/kernel/domain/test_serialization.py",
})


class TestMismatch(NamedTuple):
    """Represents a test structure mismatch."""

    test_file: str
    expected_source: str
    issue: str


def get_source_files() -> set[Path]:
    """Get all Python source files in the hexdag module.

    Returns
    -------
        Set of Path objects for source files.
    """
    src_root = Path("hexdag")
    if not src_root.exists():
        return set()

    source_files = set()
    for py_file in src_root.rglob("*.py"):
        # Skip __init__.py files and __pycache__
        if py_file.name != "__init__.py" and "__pycache__" not in str(py_file):
            source_files.add(py_file)

    return source_files


def get_test_files() -> set[Path]:
    """Get all test files in the hexdag test directory.

    Returns
    -------
        Set of Path objects for test files.
    """
    test_root = Path("tests/hexdag")
    if not test_root.exists():
        return set()

    test_files = set()
    for py_file in test_root.rglob("test_*.py"):
        test_files.add(py_file)

    return test_files


def source_to_test_path(source_file: Path) -> Path:
    """Convert a source file path to its expected test file path.

    Handles underscore-prefixed source files: ``_discovery.py`` maps to
    ``test_discovery.py`` (the leading underscore is stripped).

    Returns
    -------
        Expected test file path.
    """
    # hexdag/kernel/domain/dag.py -> tests/hexdag/kernel/domain/test_dag.py
    # hexdag/stdlib/adapters/_discovery.py -> tests/hexdag/stdlib/adapters/test_discovery.py

    # Remove the hexdag prefix
    relative_path = source_file.relative_to(Path("hexdag"))

    # Strip leading underscore from module name for test naming
    module_name = relative_path.name
    if module_name.startswith("_") and module_name != "__init__.py":
        module_name = module_name[1:]

    # Change module.py to test_module.py
    test_filename = f"test_{module_name}"
    return Path("tests/hexdag") / relative_path.parent / test_filename


def test_to_source_candidates(test_file: Path) -> list[Path]:
    """Convert a test file path to candidate source file paths.

    Returns both ``module.py`` and ``_module.py`` variants so that
    underscore-prefixed private modules (e.g. ``_discovery.py``) are
    correctly matched by ``test_discovery.py``.

    Returns
    -------
        List of candidate source file paths.

    Raises
    ------
    ValueError
        If test file doesn't follow naming convention.
    """
    # tests/hexdag/kernel/domain/test_dag.py -> hexdag/kernel/domain/dag.py

    # Remove the tests/hexdag prefix
    relative_path = test_file.relative_to(Path("tests/hexdag"))

    # Remove test_ prefix from filename
    if not relative_path.name.startswith("test_"):
        raise ValueError(f"Test file {test_file} doesn't follow test_ naming convention")

    source_filename = relative_path.name[5:]  # Remove "test_"
    parent = Path("hexdag") / relative_path.parent

    # Return both plain and underscore-prefixed variants
    return [parent / source_filename, parent / f"_{source_filename}"]


def _find_source_for_test(test_file: Path, source_files: set[Path]) -> Path | None:
    """Find the matching source file for a test file.

    Returns
    -------
        The matching source path, or None if no match found.
    """
    try:
        candidates = test_to_source_candidates(test_file)
    except ValueError:
        return None
    for candidate in candidates:
        if candidate in source_files:
            return candidate
    return None


def check_test_structure() -> list[TestMismatch]:
    """Check the test structure for consistency.

    Returns
    -------
        List of TestMismatch objects for any issues found.
    """
    source_files = get_source_files()
    test_files = get_test_files()

    mismatches = []

    # Check each test file has a corresponding source file
    test_files_with_no_source = []
    for test_file in test_files:
        # Skip explicitly allowed orphan test files
        if str(test_file) in _ALLOWED_ORPHANS:
            continue

        try:
            candidates = test_to_source_candidates(test_file)
        except ValueError as e:
            mismatches.append(
                TestMismatch(test_file=str(test_file), expected_source="N/A", issue=str(e))
            )
            continue

        if not any(c in source_files for c in candidates):
            mismatches.append(
                TestMismatch(
                    test_file=str(test_file),
                    expected_source=str(candidates[0]),
                    issue="Test file exists but corresponding source file is missing",
                )
            )
            test_files_with_no_source.append(test_file)

    # For orphaned test files, suggest they might be in wrong location
    for orphaned_test in test_files_with_no_source:
        test_module = orphaned_test.stem[5:]  # Remove "test_" prefix

        # Find source files with the same module name (plain or underscore-prefixed)
        possible_sources = [
            sf for sf in source_files if sf.stem == test_module or sf.stem == f"_{test_module}"
        ]

        if possible_sources:
            # Suggest the most likely correct location
            suggested_source = possible_sources[0]  # Take first match
            expected_test = source_to_test_path(suggested_source)

            mismatches.append(
                TestMismatch(
                    test_file=str(orphaned_test),
                    expected_source=str(suggested_source),
                    issue=f"Test file should be at {expected_test} to match source file",
                )
            )

    return mismatches


def main() -> int:
    """Check test structure and return exit code.

    Returns
    -------
        Exit code (0 for success, 1 for failure).
    """
    print("🔍 Checking test structure consistency...")

    mismatches = check_test_structure()

    if not mismatches:
        print("✅ Test structure is consistent!")
        return 0

    print(f"\n❌ Found {len(mismatches)} test structure issue(s):")
    print("=" * 80)

    for i, mismatch in enumerate(mismatches, 1):
        print(f"\n{i}. {mismatch.issue}")
        print(f"   Test file: {mismatch.test_file}")
        if mismatch.expected_source != "N/A":
            print(f"   Expected source: {mismatch.expected_source}")

    print("\n" + "=" * 80)
    print("📋 Test Structure Convention:")
    print("   Source: hexdag/path/to/module.py")
    print("   Test:   tests/hexdag/path/to/test_module.py")
    print()
    print("💡 To fix:")
    print("   1. Move test files to the correct location")
    print("   2. Remove orphaned test files")
    print("   3. Ensure test files follow test_<module>.py naming")

    return 1


if __name__ == "__main__":
    sys.exit(main())
