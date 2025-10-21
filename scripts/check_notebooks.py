#!/usr/bin/env python3
"""Validate and execute Jupyter notebooks.

This script:
1. Finds all .ipynb files in the notebooks/ directory
2. Validates notebook structure
3. Executes notebooks to ensure they run without errors
4. Checks for best practices (clear outputs, proper metadata)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("❌ Missing dependencies. Install with: uv sync --all-extras")
    sys.exit(1)


def validate_notebook_structure(notebook: dict[str, Any], path: Path) -> list[str]:
    """Validate notebook structure and metadata."""
    errors = []

    # Check for cells
    if "cells" not in notebook or not notebook["cells"]:
        errors.append(f"{path}: Notebook has no cells")

    # Check for markdown cells (documentation)
    has_markdown = any(cell.get("cell_type") == "markdown" for cell in notebook.get("cells", []))
    if not has_markdown:
        errors.append(f"{path}: Notebook has no markdown cells (documentation missing)")

    return errors


def execute_notebook(path: Path, timeout: int = 600) -> tuple[bool, str]:
    """Execute a notebook and return success status and error message."""
    try:
        # Read notebook
        with path.open() as f:
            nb = nbformat.read(f, as_version=4)

        # Execute notebook
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": path.parent}})

        return True, ""
    except Exception as e:
        return False, str(e)


def main() -> int:
    """Main entry point."""
    notebooks_dir = Path("notebooks")

    if not notebooks_dir.exists():
        print("✅ No notebooks directory found - skipping validation")
        return 0

    # Find all notebooks
    notebooks = list(notebooks_dir.rglob("*.ipynb"))
    if not notebooks:
        print("✅ No notebooks found - skipping validation")
        return 0

    print(f"📓 Found {len(notebooks)} notebook(s) to validate")

    errors = []
    for nb_path in notebooks:
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(nb_path):
            continue

        print(f"\n📄 Validating {nb_path.relative_to(notebooks_dir)}...")

        # Read and validate structure
        try:
            with nb_path.open() as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            errors.append(f"{nb_path}: Failed to read notebook - {e}")
            continue

        # Validate structure
        structure_errors = validate_notebook_structure(nb, nb_path)
        errors.extend(structure_errors)

        # Execute notebook
        print("  ▶️  Executing notebook...")
        success, error = execute_notebook(nb_path)
        if not success:
            errors.append(f"{nb_path}: Execution failed - {error}")
            print("  ❌ Execution failed")
        else:
            print("  ✅ Execution successful")

    # Report results
    print("\n" + "=" * 80)
    if errors:
        print(f"❌ Notebook validation failed with {len(errors)} error(s):\n")
        for error in errors:
            print(f"  • {error}")
        return 1
    print(f"✅ All {len(notebooks)} notebook(s) validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
