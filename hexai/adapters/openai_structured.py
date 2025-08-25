"""
Compatibility shim for test naming convention.

This module exists only to satisfy the test-structure hook.
Structured parsing lives in `hexai.adapters.openai`.
"""

from .openai import _try_parse_structured  # re-export for clarity

__all__ = ["_try_parse_structured"]
