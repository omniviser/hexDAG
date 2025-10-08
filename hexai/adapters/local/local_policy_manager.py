"""Local implementation of PolicyManagerPort using Python built-ins."""

import heapq
import weakref
from collections import defaultdict
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakSet

from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    SubscriberType,
)
from hexai.core.ports.policy_manager import Policy, PolicyManagerPort
from hexai.core.registry import adapter


@adapter(implements_port=PolicyManagerPort, name="local_policy_manager")
class LocalPolicyManager(PolicyManagerPort):
    """Local policy manager using WeakSet and heapq for efficient management.

    This implementation uses Python's built-in weak reference containers for
    automatic memory management and heapq for priority-based execution.

    Key Features:
    - WeakSet for automatic cleanup of USER/TEMPORARY policies
    - WeakKeyDictionary for metadata that cleans up with policies
    - Strong references for CORE/PLUGIN policies that shouldn't be GC'd
    - Heapq for efficient O(log n) priority queue operations
    - Type-based filtering and management
    """

    def __init__(self) -> None:
        """Initialize the local policy manager."""
        self._policies: WeakSet[Policy] = WeakSet()
        self._policy_metadata: WeakKeyDictionary[Policy, dict[str, Any]] = WeakKeyDictionary()
        self._by_type: dict[SubscriberType, WeakSet[Policy]] = defaultdict(WeakSet)

        self._strong_refs: dict[SubscriberType, set[Policy]] = {
            SubscriberType.CORE: set(),
            SubscriberType.PLUGIN: set(),
        }

        self._subscriptions: weakref.WeakValueDictionary[str, Policy] = (
            weakref.WeakValueDictionary()
        )

        self._priority_queue_cache: list[tuple[int, int, Policy]] | None = None
        self._cache_generation: int = 0

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate all policies in priority order.

        Uses a cached heap-based priority queue for efficient ordering.
        First non-PROCEED response wins (veto pattern).

        Args:
        ----
            context: Policy evaluation context.

        Returns:
        -------
            PolicyResponse with signal and optional data
        """
        if self._priority_queue_cache is None:
            pq: list[tuple[int, int, Policy]] = []
            for idx, policy in enumerate(self._policies):
                if policy in self._policy_metadata:
                    metadata = self._policy_metadata[policy]
                    priority = metadata.get("priority", policy.priority)
                    heapq.heappush(pq, (priority, idx, policy))
            self._priority_queue_cache = pq
        else:
            pq = self._priority_queue_cache.copy()

        while pq:
            _, _, policy = heapq.heappop(pq)
            try:
                response = await policy.evaluate(context)
                if response.signal != PolicySignal.PROCEED:
                    return response
            except Exception:
                continue  # nosec B112

        return PolicyResponse(signal=PolicySignal.PROCEED)

    def subscribe(
        self, policy: Policy, subscriber_type: SubscriberType = SubscriberType.USER
    ) -> str:
        """Subscribe a policy with automatic reference management.

        CORE and PLUGIN policies are kept with strong references.
        USER and TEMPORARY policies use weak references for auto-cleanup.

        Args:
        ----
            policy: Policy to subscribe.
            subscriber_type: Type of subscriber.

        Returns:
        -------
            Subscription ID for unsubscribe
        """
        subscription_id = str(uuid4())

        self._policies.add(policy)

        self._policy_metadata[policy] = {
            "subscription_id": subscription_id,
            "subscriber_type": subscriber_type,
            "priority": getattr(policy, "priority", 100),
        }

        self._by_type[subscriber_type].add(policy)

        if subscriber_type in (SubscriberType.CORE, SubscriberType.PLUGIN):
            self._strong_refs[subscriber_type].add(policy)

        self._subscriptions[subscription_id] = policy
        self._priority_queue_cache = None

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe a policy by ID.

        Args:
        ----
            subscription_id: ID returned from subscribe.

        Returns:
        -------
            True if unsubscribed, False if not found
        """
        policy = self._subscriptions.get(subscription_id)
        if policy is None:
            return False

        metadata = self._policy_metadata.get(policy, {})
        subscriber_type = metadata.get("subscriber_type")

        self._policies.discard(policy)

        if subscriber_type:
            self._by_type[subscriber_type].discard(policy)
            if subscriber_type in self._strong_refs:
                self._strong_refs[subscriber_type].discard(policy)

        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]

        self._priority_queue_cache = None

        return True

    def clear(self, subscriber_type: SubscriberType | None = None) -> None:
        """Clear policies, optionally by type.

        Args:
        ----
            subscriber_type: If specified, only clear this type.
        """
        if subscriber_type is None:
            self._policies.clear()
            self._policy_metadata.clear()
            self._by_type.clear()
            for refs in self._strong_refs.values():
                refs.clear()
            self._subscriptions.clear()
            self._priority_queue_cache = None
        else:
            policies_to_remove = list(self._by_type[subscriber_type])
            for policy in policies_to_remove:
                metadata = self._policy_metadata.get(policy, {})
                sub_id = metadata.get("subscription_id")
                if sub_id:
                    self.unsubscribe(sub_id)

    def get_subscriptions(self) -> dict[str, tuple[Policy, SubscriberType]]:
        """Get all active subscriptions.

        Returns:
        -------
            Dict of subscription_id -> (policy, subscriber_type)
        """
        result = {}
        for sub_id, policy in self._subscriptions.items():
            metadata = self._policy_metadata.get(policy, {})
            subscriber_type = metadata.get("subscriber_type", SubscriberType.USER)
            result[sub_id] = (policy, subscriber_type)
        return result

    def get_policies_by_type(self, subscriber_type: SubscriberType) -> list[Policy]:
        """Get policies for a specific subscriber type.

        Args:
        ----
            subscriber_type: Type to filter by.

        Returns:
        -------
            List of policies of that type
        """
        return list(self._by_type[subscriber_type])

    def __len__(self) -> int:
        """Return count of active policies."""
        return len(self._policies)

    def __contains__(self, policy: Policy) -> bool:
        """Check if policy is registered."""
        return policy in self._policies
