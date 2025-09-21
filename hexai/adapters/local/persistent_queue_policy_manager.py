"""Policy manager with persistent priority queue for high-frequency evaluations."""

import heapq
from typing import Any
from uuid import uuid4

from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    SubscriberType,
)
from hexai.core.ports.policy_manager import Policy, PolicyManagerPort
from hexai.core.registry import adapter


@adapter(implements_port=PolicyManagerPort, name="persistent_queue_policy_manager")
class PersistentQueuePolicyManager(PolicyManagerPort):
    """Policy manager maintaining a persistent priority queue.

    This implementation is optimized for scenarios with:
    - Frequent policy evaluations
    - Relatively stable policy set
    - Performance-critical evaluation paths

    Trade-offs:
    - More complex than LocalPolicyManager
    - No automatic garbage collection (all references are strong)
    - Must manually maintain heap invariant
    """

    def __init__(self) -> None:
        """Initialize the persistent queue policy manager."""
        # Persistent priority queue
        self._queue: list[tuple[int, int, str, Policy]] = []
        self._counter = 0  # For stable sorting

        # Policy registry
        self._policies: dict[str, Policy] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

        # Subscription tracking
        self._dirty = False  # Flag to rebuild queue if needed

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate policies using the persistent queue.

        Args:
        ----
            context: Policy evaluation context.

        Returns:
        -------
            PolicyResponse with signal and optional data.
        """
        # Rebuild queue if dirty (after subscribe/unsubscribe)
        if self._dirty:
            self._rebuild_queue()

        # Create a copy to iterate (preserves queue)
        queue_copy = list(self._queue)

        # Process in priority order
        for _, _, sub_id, policy in sorted(queue_copy):
            # Skip if policy was removed (defensive)
            if sub_id not in self._policies:
                continue

            try:
                response = await policy.evaluate(context)
                if response.signal != PolicySignal.PROCEED:
                    return response
            except Exception:
                # Skip failed policies - intentional behavior
                continue  # nosec B112

        return PolicyResponse(signal=PolicySignal.PROCEED)

    def subscribe(
        self, policy: Policy, subscriber_type: SubscriberType = SubscriberType.USER
    ) -> str:
        """Subscribe a policy to the manager.

        Args:
        ----
            policy: Policy to subscribe.
            subscriber_type: Type of subscriber.

        Returns:
        -------
            Subscription ID for later unsubscribe.
        """
        subscription_id = str(uuid4())

        # Store policy and metadata
        self._policies[subscription_id] = policy
        self._metadata[subscription_id] = {
            "subscriber_type": subscriber_type,
            "priority": getattr(policy, "priority", 100),
        }

        # Add to queue
        priority = self._metadata[subscription_id]["priority"]
        heapq.heappush(self._queue, (priority, self._counter, subscription_id, policy))
        self._counter += 1

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe a policy by subscription ID.

        Args:
        ----
            subscription_id: ID returned from subscribe.

        Returns:
        -------
            True if unsubscribed, False if not found.
        """
        if subscription_id not in self._policies:
            return False

        # Remove from registry
        del self._policies[subscription_id]
        del self._metadata[subscription_id]

        # Mark queue as dirty (needs rebuild)
        self._dirty = True

        return True

    def clear(self, subscriber_type: SubscriberType | None = None) -> None:
        """Clear subscribed policies.

        Args:
        ----
            subscriber_type: If specified, only clear this type.
        """
        if subscriber_type is None:
            # Clear everything
            self._policies.clear()
            self._metadata.clear()
            self._queue.clear()
            self._counter = 0
            self._dirty = False
        else:
            # Clear specific type
            to_remove = [
                sub_id
                for sub_id, meta in self._metadata.items()
                if meta["subscriber_type"] == subscriber_type
            ]
            for sub_id in to_remove:
                del self._policies[sub_id]
                del self._metadata[sub_id]

            if to_remove:
                self._dirty = True

    def get_subscriptions(self) -> dict[str, tuple[Policy, SubscriberType]]:
        """Get all active subscriptions.

        Returns:
        -------
            Dict of subscription_id -> (policy, subscriber_type).
        """
        return {
            sub_id: (policy, self._metadata[sub_id]["subscriber_type"])
            for sub_id, policy in self._policies.items()
        }

    def get_policies_by_type(self, subscriber_type: SubscriberType) -> list[Policy]:
        """Get policies for a specific subscriber type.

        Args:
        ----
            subscriber_type: Type to filter by.

        Returns:
        -------
            List of policies of that type.
        """
        return [
            policy
            for sub_id, policy in self._policies.items()
            if self._metadata[sub_id]["subscriber_type"] == subscriber_type
        ]

    def _rebuild_queue(self) -> None:
        """Rebuild the priority queue from current policies."""
        self._queue.clear()
        self._counter = 0

        for sub_id, policy in self._policies.items():
            priority = self._metadata[sub_id]["priority"]
            heapq.heappush(self._queue, (priority, self._counter, sub_id, policy))
            self._counter += 1

        self._dirty = False

    def __len__(self) -> int:
        """Return count of active policies."""
        return len(self._policies)

    def __contains__(self, policy: Policy) -> bool:
        """Check if policy is registered."""
        return any(p is policy for p in self._policies.values())
