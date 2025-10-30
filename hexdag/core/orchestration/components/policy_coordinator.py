"""Policy coordinator for orchestrator execution control.

This module provides the PolicyCoordinator class that handles policy evaluation
and observer notifications during DAG execution.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort
    from hexdag.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexdag.core.exceptions import OrchestratorError
from hexdag.core.orchestration.models import NodeExecutionContext
from hexdag.core.orchestration.policies.models import PolicyContext, PolicyResponse, PolicySignal


class PolicyCoordinator:
    """Handles policy evaluation and observer notifications.

    This component is responsible for coordinating between the orchestrator,
    policy managers, and observer managers during DAG execution. It provides
    a clean interface for:

    - Notifying observers of execution events
    - Evaluating policies to control execution flow
    - Checking policy signals and handling control flow

    Single Responsibility: Coordinate policy decisions and event observation.

    Examples
    --------
    Example usage::

        coordinator = PolicyCoordinator()
        await coordinator.notify_observer(observer_manager, NodeStarted(...))
        response = await coordinator.evaluate_policy(policy_manager, event, context)
        coordinator.check_policy_signal(response, "Node execution")
    """

    async def notify_observer(
        self, observer_manager: ObserverManagerPort | None, event: Any
    ) -> None:
        """Notify observer manager of an event if it exists.

        Parameters
        ----------
        observer_manager : ObserverManagerPort | None
            Observer manager to notify (None if no observer configured)
        event : Any
            Event to send (typically NodeStarted, NodeCompleted, etc.)

        Examples
        --------
        Example usage::

            from hexdag.core.orchestration.events import NodeStarted
            event = NodeStarted(name="my_node", wave_index=0)
            await coordinator.notify_observer(observer_manager, event)
        """
        if observer_manager:
            await observer_manager.notify(event)

    async def evaluate_policy(
        self,
        policy_manager: PolicyManagerPort | None,
        event: Any,
        context: NodeExecutionContext,
        node_id: str | None = None,
        wave_index: int | None = None,
        attempt: int = 1,
    ) -> PolicyResponse:
        """Evaluate policy for an event and return the policy decision.

        Creates a PolicyContext from the execution context and event, then
        evaluates the policy using the policy manager (if configured).

        Parameters
        ----------
        policy_manager : PolicyManagerPort | None
            Policy manager to evaluate (None if no policy configured)
        event : Any
            Event triggering policy evaluation
        context : NodeExecutionContext
            Current execution context
        node_id : str | None
            Optional node ID override (uses context.node_id if None)
        wave_index : int | None
            Optional wave index override (uses context.wave_index if None)
        attempt : int
            Attempt number for retries (default: 1)

        Returns
        -------
        PolicyResponse
            Policy decision (default: PROCEED if no policy manager)

        Examples
        --------
        Example usage::

            response = await coordinator.evaluate_policy(
                policy_manager=my_policy,
                event=NodeStarted(name="test"),
                context=execution_context,
                node_id="my_node",
                wave_index=0
            )
            if response.signal == PolicySignal.SKIP:
                print("Node was skipped by policy")
        """
        policy_context = PolicyContext(
            event=event,
            dag_id=context.dag_id,
            node_id=node_id or context.node_id,
            wave_index=wave_index or context.wave_index,
            attempt=attempt or context.attempt,
        )

        if policy_manager:
            return await policy_manager.evaluate(policy_context)
        return PolicyResponse()  # Default: proceed

    def check_policy_signal(self, response: PolicyResponse, context: str) -> None:
        """Check policy signal and raise error if not PROCEED.

        This is a convenience method for enforcing policy decisions that
        should block execution. If the policy signal is not PROCEED,
        raises an OrchestratorError with a descriptive message.

        Parameters
        ----------
        response : PolicyResponse
            Policy response to check
        context : str
            Context description for error message (e.g., "Pipeline start")

        Raises
        ------
        OrchestratorError
            If policy signal is not PROCEED

        Examples
        --------
        Example usage::

            response = await coordinator.evaluate_policy(...)
            coordinator.check_policy_signal(response, "Pipeline start")
            # Raises OrchestratorError if signal != PROCEED
        """
        if response.signal != PolicySignal.PROCEED:
            raise OrchestratorError(f"{context} blocked: {response.signal.value}")
