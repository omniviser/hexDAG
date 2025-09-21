"""Local adapter implementations that run in-process."""

from .in_memory_memory import InMemoryMemory
from .local_policy_manager import LocalPolicyManager

__all__ = [
    "InMemoryMemory",
    "LocalPolicyManager",
]
