"""Policy Manager Port - Clean interface for execution control policies."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from hexai.core.registry import port


class PolicySignal(Enum):
    """Execution control signals."""

    PROCEED = "proceed"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    FAIL = "fail"


@dataclass
class PolicyContext:
    """Context for policy evaluation."""

    dag_id: str
    node_id: str | None = None
    attempt: int = 1
    error: Exception | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PolicyResponse:
    """Policy evaluation response."""

    signal: PolicySignal = PolicySignal.PROCEED
    data: Any = None
    metadata: dict[str, Any] | None = None


class SubscriberType(Enum):
    """Types of policy subscribers for categorization."""

    CORE = "core"  # Built-in framework policies
    PLUGIN = "plugin"  # Plugin-provided policies
    USER = "user"  # User-defined policies
    TEMPORARY = "temporary"  # Temporary/test policies


class Policy(Protocol):
    """Protocol for individual policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for identification."""
        ...

    @property
    def priority(self) -> int:
        """Priority (lower = first). Default: 100."""
        return 100

    @abstractmethod
    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate policy and return decision."""
        ...


@port(name="policy_manager", namespace="core")
class PolicyManagerPort(Protocol):
    """Port for managing execution control policies.

    Provides subscription-based policy management with automatic cleanup
    using weak references and categorization by subscriber type.
    """

    @abstractmethod
    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate all subscribed policies for the given context.

        Args:
            context: Evaluation context with execution details

        Returns:
            Aggregated policy response (first non-PROCEED or PROCEED)
        """
        ...

    @abstractmethod
    def subscribe(
        self, policy: Policy, subscriber_type: SubscriberType = SubscriberType.USER
    ) -> str:
        """Subscribe a policy to the manager.

        Args:
            policy: Policy instance to subscribe
            subscriber_type: Category of the policy subscriber

        Returns:
            Subscription ID for later unsubscribe
        """
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe a policy by subscription ID.

        Args:
            subscription_id: ID returned from subscribe

        Returns:
            True if unsubscribed, False if not found
        """
        ...

    @abstractmethod
    def clear(self, subscriber_type: SubscriberType | None = None) -> None:
        """Clear subscribed policies.

        Args:
            subscriber_type: If specified, only clear policies of this type
        """
        ...

    @abstractmethod
    def get_subscriptions(self) -> dict[str, tuple[Policy, SubscriberType]]:
        """Get all active subscriptions.

        Returns:
            Dict of subscription_id -> (policy, subscriber_type)
        """
        ...
