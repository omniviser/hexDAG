"""Local implementation of PolicyManagerPort using Python built-ins."""

import asyncio
import heapq
import inspect
import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from typing import Annotated, Any, cast, get_args, get_origin
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakSet

from pydantic import BaseModel, ConfigDict, field_validator

from hexai.core.application.events.decorators import (
    EVENT_METADATA_ATTR,
    EventDecoratorMetadata,
    EventType,
    EventTypesInput,
    normalize_event_types,
)
from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    SubscriberType,
)
from hexai.core.ports.policy_manager import Policy, PolicyManagerPort
from hexai.core.registry import adapter

PolicyFunc = Callable[..., PolicyResponse]
AsyncPolicyFunc = Callable[..., Awaitable[PolicyResponse]]


class PolicyRegistrationConfig(BaseModel):
    """Validated configuration for policy registration."""

    model_config = ConfigDict(extra="forbid")

    priority: int = 100
    name: str | None = None
    description: str | None = None
    event_types: set[EventType] | None = None
    keep_alive: bool = False
    subscriber_type: SubscriberType = SubscriberType.USER

    @field_validator("event_types", mode="before")
    @classmethod
    def validate_event_types(cls, value: EventTypesInput) -> set[EventType] | None:
        return normalize_event_types(value)


def _get_policy_metadata(handler: Any) -> EventDecoratorMetadata | None:
    """Return policy metadata if the handler is decorated."""

    metadata = getattr(handler, EVENT_METADATA_ATTR, None)
    if isinstance(metadata, EventDecoratorMetadata) and metadata.kind == "control_handler":
        return metadata
    return None


def _ensure_policy_response_return_type(func: Callable[..., Any]) -> None:
    """Ensure decorated functions declare PolicyResponse as return type."""

    func_name = getattr(func, "__name__", repr(func))
    annotations = getattr(func, "__annotations__", {}) or {}
    return_type = annotations.get("return")

    if return_type is None:
        raise TypeError(f"Policy handler {func_name} must declare PolicyResponse as return type")

    if _is_policy_response_annotation(return_type):
        return

    raise TypeError(f"Policy handler {func_name} must return PolicyResponse, got {return_type}")


def _is_policy_response_annotation(annotation: Any) -> bool:
    """Check whether an annotation represents a PolicyResponse."""

    if annotation is PolicyResponse:
        return True

    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        valid = {
            PolicyResponse.__name__,
            f"Awaitable[{PolicyResponse.__name__}]",
            f"Coroutine[Any,Any,{PolicyResponse.__name__}]",
        }
        return normalized in valid

    origin = get_origin(annotation)
    if origin is None:
        return False

    if origin is Annotated:
        args = get_args(annotation)
        return bool(args) and _is_policy_response_annotation(args[0])

    if origin in {Coroutine, asyncio.Future}:
        args = get_args(annotation)
        if len(args) == 3:
            return _is_policy_response_annotation(args[2])
        return False

    if origin in {Awaitable, asyncio.Future}:
        args = get_args(annotation)
        if not args:
            return False
        return _is_policy_response_annotation(args[0])

    return False


class FunctionPolicy(Policy):
    """Wrapper that adapts simple functions to the Policy interface."""

    def __init__(self, func: PolicyFunc | AsyncPolicyFunc, name: str, priority: int) -> None:
        self._func: PolicyFunc | AsyncPolicyFunc = func
        self._name = name
        self._priority = priority
        self._call_mode = self._determine_call_mode(func)

    @staticmethod
    def _determine_call_mode(func: Callable[..., Any]) -> str:
        signature = inspect.signature(func)
        params = [
            param
            for param in signature.parameters.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]

        if len(params) == 1:
            return "context"
        if len(params) == 2:
            return "event_context"

        raise TypeError("Policy functions must accept either (context) or (event, context).")

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        if self._call_mode == "context":
            call_result = self._func(context)
        else:
            call_result = self._func(context.event, context)

        if isinstance(call_result, Awaitable):
            result: Any = await call_result
        elif inspect.isawaitable(call_result):
            result = await cast("Awaitable[PolicyResponse]", call_result)
        else:
            result = call_result

        if not isinstance(result, PolicyResponse):
            message = (
                f"Policy handler {self._name} must return PolicyResponse, got "
                f"{type(result).__name__}"
            )
            raise TypeError(message)

        return result


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
        # Core storage using weak references
        self._policies: WeakSet[Policy] = WeakSet()

        # Metadata using WeakKeyDictionary for automatic cleanup
        self._policy_metadata: WeakKeyDictionary[Policy, dict[str, Any]] = WeakKeyDictionary()

        # Subscription tracking by type for efficient filtering
        self._by_type: dict[SubscriberType, WeakSet[Policy]] = defaultdict(WeakSet)

        # Strong references when policies must stay alive (CORE/PLUGIN or keep_alive)
        self._strong_refs: dict[SubscriberType, set[Policy]] = defaultdict(set)

        # Subscription ID mapping
        self._subscriptions: weakref.WeakValueDictionary[str, Policy] = (
            weakref.WeakValueDictionary()
        )

        # Quick lookup by policy name to prevent duplicates
        self._name_index: dict[str, str] = {}

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate all policies in priority order.

        Uses a heap-based priority queue for efficient ordering.
        First non-PROCEED response wins (veto pattern).

        Args:
        ----
            context: Policy evaluation context.

        Returns:
        -------
            PolicyResponse with signal and optional data
        """
        pq: list[tuple[int, int, Policy]] = []

        for idx, policy in enumerate(list(self._policies)):
            metadata = self._policy_metadata.get(policy)
            if metadata is None:
                continue

            event_types = metadata.get("event_types")
            if event_types:
                event = getattr(context, "event", None)
                if event is None or not isinstance(event, tuple(event_types)):
                    continue

            priority = metadata.get("priority", getattr(policy, "priority", 100))
            heapq.heappush(pq, (priority, idx, policy))

        # Process in priority order
        while pq:
            _, _, policy = heapq.heappop(pq)
            try:
                response = await policy.evaluate(context)
                if response.signal != PolicySignal.PROCEED:
                    return response
            except Exception:
                # Skip failed policies - this is intentional behavior
                continue  # nosec B112

        return PolicyResponse(signal=PolicySignal.PROCEED)

    def register(
        self,
        policy: Policy | PolicyFunc | AsyncPolicyFunc,
        *,
        priority: int | None = None,
        name: str | None = None,
        event_types: EventTypesInput = None,
        description: str | None = None,
        keep_alive: bool = False,
        subscriber_type: SubscriberType = SubscriberType.USER,
    ) -> str:
        """Register a policy handler (class-based or function-based)."""

        metadata = _get_policy_metadata(policy)

        resolved_priority = (
            priority
            if priority is not None
            else (
                metadata.priority
                if metadata and metadata.priority is not None
                else getattr(policy, "priority", 100)
            )
        )

        resolved_name = (
            name if name is not None else (metadata.name if metadata and metadata.name else None)
        )

        resolved_description = (
            description
            if description is not None
            else (metadata.description if metadata and metadata.description else None)
        )

        event_types_input = (
            event_types if event_types is not None else (metadata.event_types if metadata else None)
        )
        normalized_event_types = (
            normalize_event_types(event_types_input) if event_types_input is not None else None
        )

        config = PolicyRegistrationConfig(
            priority=resolved_priority,
            name=resolved_name,
            description=resolved_description,
            event_types=normalized_event_types,
            keep_alive=keep_alive,
            subscriber_type=subscriber_type,
        )

        policy_name = config.name
        if not policy_name:
            fallback = getattr(policy, "name", None) or getattr(policy, "__name__", None)
            policy_name = fallback or f"policy_{id(policy)}"

        if policy_name in self._name_index:
            raise ValueError(f"Policy '{policy_name}' is already registered")

        keep_alive_flag = config.keep_alive

        evaluate_callable = getattr(policy, "evaluate", None)

        if callable(evaluate_callable):
            policy_obj = cast("Policy", policy)
        elif callable(policy):
            _ensure_policy_response_return_type(policy)
            policy_obj = FunctionPolicy(policy, policy_name, config.priority)
            keep_alive_flag = True
        else:
            raise TypeError("Policy must implement the Policy protocol or be a callable function")

        if policy_obj in self._policy_metadata:
            raise ValueError("Policy handler is already registered")

        subscription_id = str(uuid4())

        self._policies.add(policy_obj)
        self._policy_metadata[policy_obj] = {
            "subscription_id": subscription_id,
            "subscriber_type": config.subscriber_type,
            "priority": config.priority,
            "event_types": normalized_event_types,
            "name": policy_name,
            "description": config.description,
            "keep_alive": keep_alive_flag,
        }

        self._by_type[config.subscriber_type].add(policy_obj)

        if keep_alive_flag or config.subscriber_type in (
            SubscriberType.CORE,
            SubscriberType.PLUGIN,
        ):
            self._strong_refs[config.subscriber_type].add(policy_obj)

        self._subscriptions[subscription_id] = policy_obj
        self._name_index[policy_name] = subscription_id

        return subscription_id

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

        keep_alive = subscriber_type in (SubscriberType.CORE, SubscriberType.PLUGIN)
        return self.register(
            policy,
            subscriber_type=subscriber_type,
            keep_alive=keep_alive,
        )

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

        # Get metadata before removal
        metadata = self._policy_metadata.get(policy, {})
        subscriber_type = metadata.get("subscriber_type")
        policy_name = metadata.get("name")

        # Remove from all collections
        self._policies.discard(policy)
        self._policy_metadata.pop(policy, None)

        if policy_name:
            self._name_index.pop(policy_name, None)

        if subscriber_type:
            self._by_type[subscriber_type].discard(policy)
            self._strong_refs[subscriber_type].discard(policy)

        # Remove subscription
        self._subscriptions.pop(subscription_id, None)

        return True

    def clear(self, subscriber_type: SubscriberType | None = None) -> None:
        """Clear policies, optionally by type.

        Args:
        ----
            subscriber_type: If specified, only clear this type.
        """
        if subscriber_type is None:
            # Clear all
            self._policies.clear()
            self._policy_metadata.clear()
            self._by_type.clear()
            self._strong_refs.clear()
            self._subscriptions.clear()
            self._name_index.clear()
        else:
            # Clear specific type
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
