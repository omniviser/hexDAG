"""Local adapter implementations that run in-process."""

from .local_observer_manager import LocalObserverManager
from .local_policy_manager import LocalPolicyManager

__all__ = [
    "LocalObserverManager",
    "LocalPolicyManager",
]
