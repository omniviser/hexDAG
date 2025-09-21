"""Policy Manager Port - Clean interface for execution control policies."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    SubscriberType,
)
from hexai.core.registry import port


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


@runtime_checkable
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

    @abstractmethod
    def get_policies_by_type(self, subscriber_type: SubscriberType) -> list[Policy]:
        """Get all policies of a specific subscriber type.

        Args:
            subscriber_type: Type to filter by

        Returns:
            List of policies of that type
        """
        ...

    def count(self) -> int:
        """Get the count of active policies.

        Returns:
            Number of active policies

        Note:
            Implementations should also support len() via __len__ for Pythonic usage
        """
        return len(self.get_subscriptions())
