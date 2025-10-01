#!/usr/bin/env python3
"""
Pre-commit hook to validate test structure consistency.

This script ensures that:
1. Every test file corresponds to an existing source file
2. Test files are in the correct location relative to their source files
3. No orphaned test files exist
4. Test files follow the naming convention: test_<module_name>.py

Expected structure:
- Source: hexai/path/to/module.py
- Test:   tests/hexai/path/to/test_module.py
"""

import sys
from pathlib import Path
from typing import NamedTuple


class TestMismatch(NamedTuple):
    """Represents a test structure mismatch."""

    test_file: str
    expected_source: str
    issue: str


def get_source_files() -> set[Path]:
    """Get all Python source files in the hexai module.

    Returns
    -------
        Set of Path objects for source files.
    """
    src_root = Path("hexai")
    if not src_root.exists():
        return set()

    source_files = set()
    for py_file in src_root.rglob("*.py"):
        # Skip __init__.py files and __pycache__
        if py_file.name != "__init__.py" and "__pycache__" not in str(py_file):
            source_files.add(py_file)

    return source_files


def get_test_files() -> set[Path]:
    """Get all test files in the hexai test directory.

    Returns
    -------
        Set of Path objects for test files.
    """
    test_root = Path("tests/hexai")
    if not test_root.exists():
        return set()

    test_files = set()
    for py_file in test_root.rglob("test_*.py"):
        test_files.add(py_file)

    return test_files


def source_to_test_path(source_file: Path) -> Path:
    """Convert a source file path to its expected test file path.

    Returns
    -------
        Expected test file path.
    """
    # hexai/core/domain/dag.py -> tests/hexai/core/domain/test_dag.py

    # Remove the hexai prefix
    relative_path = source_file.relative_to(Path("hexai"))

    # Change module.py to test_module.py
    test_filename = f"test_{relative_path.name}"
    return Path("tests/hexai") / relative_path.parent / test_filename


def test_to_source_path(test_file: Path) -> Path:
    """Convert a test file path to its expected source file path.

    Returns
    -------
        Expected source file path.

    Raises
    ------
    ValueError
        If test file doesn't follow naming convention.
    """
    # tests/hexai/core/domain/test_dag.py -> hexai/core/domain/dag.py

    # Remove the tests/hexai prefix
    relative_path = test_file.relative_to(Path("tests/hexai"))

    # Remove test_ prefix from filename
    if not relative_path.name.startswith("test_"):
        raise ValueError(f"Test file {test_file} doesn't follow test_ naming convention")

    source_filename = relative_path.name[5:]  # Remove "test_"
    return Path("hexai") / relative_path.parent / source_filename


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
    for test_file in test_files:
        try:
            expected_source = test_to_source_path(test_file)

            if expected_source not in source_files:
                mismatches.append(
                    TestMismatch(
                        test_file=str(test_file),
                        expected_source=str(expected_source),
                        issue="Test file exists but corresponding source file is missing",
                    )
                )
        except ValueError as e:
            mismatches.append(
                TestMismatch(test_file=str(test_file), expected_source="N/A", issue=str(e))
            )

    # Check if any test files are in unexpected locations
    # Only flag as wrong if the test file doesn't correspond to any existing source file
    test_files_with_no_source = []
    for test_file in test_files:
        try:
            expected_source = test_to_source_path(test_file)
            if expected_source not in source_files:
                # This test file doesn't have a corresponding source file
                test_files_with_no_source.append(test_file)
        except ValueError:
            # Already handled above in the first loop
            pass

    # For orphaned test files, suggest they might be in wrong location
    for orphaned_test in test_files_with_no_source:
        test_module = orphaned_test.stem[5:]  # Remove "test_" prefix

        # Find source files with the same module name
        possible_sources = [sf for sf in source_files if sf.stem == test_module]

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
    print("   Source: hexai/path/to/module.py")
    print("   Test:   tests/hexai/path/to/test_module.py")
    print()
    print("💡 To fix:")
    print("   1. Move test files to the correct location")
    print("   2. Remove orphaned test files")
    print("   3. Ensure test files follow test_<module>.py naming")

    return 1


if __name__ == "__main__":
    sys.exit(main())
