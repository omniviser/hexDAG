"""Mutmut configuration for mutation testing.

This file configures mutation testing for hexDAG to find weak tests.
"""

from typing import Any


def pre_mutation(context: Any) -> None:
    """Hook called before each mutation.

    Can be used to skip certain mutations or add custom logic.
    """
    # Skip mutations in test files
    if "test_" in context.filename:
        context.skip = True

    # Skip mutations in __init__.py files
    if context.filename.endswith("__init__.py"):
        context.skip = True

    # Skip mutations in CLI (tested manually)
    if "/cli/" in context.filename:
        context.skip = True
