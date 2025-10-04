"""Vulture whitelist for false positives.

This file contains imports and names that vulture incorrectly identifies as unused.
Each entry should have a comment explaining why it's needed.
"""

# Used in cast() string literal for type checking (line 467 in registry.py)
from collections.abc import Callable as CallableType

# Dummy usage to satisfy vulture
_ = CallableType
