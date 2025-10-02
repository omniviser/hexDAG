#!/usr/bin/env python3
"""Script to update all logging imports to use centralized logging."""

import re
from pathlib import Path


def update_file(file_path: Path) -> tuple[bool, str]:
    """Update a single file to use centralized logging.

    Returns
    -------
    tuple[bool, str]
        (was_modified, message)
    """
    content = file_path.read_text()
    original_content = content

    # Check if already using centralized logging
    if "from hexai.core.logging import get_logger" in content:
        return False, "Already using centralized logging"

    # Check if file uses logging
    if "logging.getLogger" not in content:
        return False, "No logging usage found"

    # Pattern 1: Remove standalone `import logging` if only used for getLogger
    # Check if logging is only used for getLogger
    logging_usage = re.findall(r"\blogging\.(\w+)", content)
    if set(logging_usage) == {"getLogger"} or set(logging_usage) <= {"getLogger", "Logger"}:
        # Remove import logging line
        content = re.sub(r"^import logging\n", "", content, flags=re.MULTILINE)

    # Pattern 2: Add centralized logging import
    # Find the last import statement
    import_lines = []
    for i, line in enumerate(content.split("\n")):
        if line.startswith("import ") or line.startswith("from "):
            import_lines.append(i)

    if import_lines:
        lines = content.split("\n")
        last_import_idx = max(import_lines)

        # Insert after last import
        lines.insert(last_import_idx + 1, "from hexai.core.logging import get_logger")
        content = "\n".join(lines)

    # Pattern 3: Replace logging.getLogger(__name__) with get_logger(__name__)
    content = re.sub(r"logging\.getLogger\(__name__\)", "get_logger(__name__)", content)

    # Pattern 4: Replace other logging.getLogger patterns
    content = re.sub(r"logging\.getLogger\(([^)]+)\)", r"get_logger(\1)", content)

    if content != original_content:
        file_path.write_text(content)
        return True, "Updated successfully"

    return False, "No changes needed"


def main() -> None:
    """Update all Python files in hexai/."""
    hexai_dir = Path(__file__).parent.parent / "hexai"

    python_files = list(hexai_dir.rglob("*.py"))

    # Exclude the logging.py module itself
    python_files = [f for f in python_files if f.name != "logging.py"]

    updated_count = 0
    skipped_count = 0

    for file_path in python_files:
        was_modified, message = update_file(file_path)

        if was_modified:
            print(f"✓ {file_path.relative_to(hexai_dir)}: {message}")
            updated_count += 1
        else:
            skipped_count += 1

    print("\nSummary:")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {skipped_count} files")


if __name__ == "__main__":
    main()
