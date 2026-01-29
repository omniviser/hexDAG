"""Execution coordinator for policy evaluation and input mapping.

This module consolidates policy evaluation and input data mapping into a single
component that handles execution coordination:

- Observer notifications during execution
- Policy evaluation for control flow
- Input preparation and dependency mapping

The ExecutionCoordinator replaces the separate PolicyCoordinator and InputMapper
components with a unified interface.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort
    from hexdag.core.ports.policy_manager import PolicyManagerPort
else:
    ObserverManagerPort = Any
    PolicyManagerPort = Any

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.exceptions import OrchestratorError
from hexdag.core.orchestration.models import NodeExecutionContext
from hexdag.core.orchestration.policies.models import PolicyContext, PolicyResponse, PolicySignal

__all__ = ["ExecutionCoordinator"]


class ExecutionCoordinator:
    """Coordinates execution context: policy evaluation, input mapping, event notification.

    This component consolidates two related responsibilities:

    1. **Policy Coordination**: Evaluating policies and notifying observers during
       DAG execution. This includes control flow decisions (skip, retry, halt).

    2. **Input Mapping**: Preparing input data for nodes based on their dependencies.
       Uses a smart mapping strategy:
       - No dependencies → initial input
       - Single dependency → pass through that result
       - Multiple dependencies → dict of results

    Examples
    --------
    Basic usage::

        coordinator = ExecutionCoordinator()

        # Notify observer of an event
        await coordinator.notify_observer(observer_manager, NodeStarted(...))

        # Evaluate policy for control flow
        response = await coordinator.evaluate_policy(
            policy_manager, event, context
        )
        coordinator.check_policy_signal(response, "Node execution")

        # Prepare input for a node
        input_data = coordinator.prepare_node_input(
            node_spec, node_results, initial_input
        )
    """

    # ========================================================================
    # Observer Notifications (from PolicyCoordinator)
    # ========================================================================

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
        >>> from hexdag.core.orchestration.events import NodeStarted
        >>> event = NodeStarted(name="my_node", wave_index=0)
        >>> await coordinator.notify_observer(observer_manager, event)  # doctest: +SKIP
        """
        if observer_manager:
            await observer_manager.notify(event)

    # ========================================================================
    # Policy Evaluation (from PolicyCoordinator)
    # ========================================================================

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
        >>> response = await coordinator.evaluate_policy(  # doctest: +SKIP
        ...     policy_manager=my_policy,
        ...     event=NodeStarted(name="test"),
        ...     context=execution_context,
        ...     node_id="my_node",
        ...     wave_index=0
        ... )
        >>> if response.signal == PolicySignal.SKIP:  # doctest: +SKIP
        ...     print("Node was skipped by policy")
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
        >>> response = await coordinator.evaluate_policy(...)  # doctest: +SKIP
        >>> coordinator.check_policy_signal(response, "Pipeline start")  # doctest: +SKIP
        """
        if response.signal != PolicySignal.PROCEED:
            raise OrchestratorError(f"{context} blocked: {response.signal.value}")

    # ========================================================================
    # Input Mapping (from InputMapper)
    # ========================================================================

    def prepare_node_input(
        self, node_spec: NodeSpec, node_results: dict[str, Any], initial_input: Any
    ) -> Any:
        """Prepare input data for node execution with simplified data mapping.

        The mapping strategy is:
        1. **No dependencies** → initial_input (entry point)
        2. **Single dependency** → results[dependency_name] (pass-through)
        3. **Multiple dependencies** → {dep1: result1, dep2: result2, ...} (namespace)

        This approach balances simplicity (pass-through for single deps) with
        clarity (named dict for multiple deps).

        Parameters
        ----------
        node_spec : NodeSpec
            Node specification containing dependencies
        node_results : dict[str, Any]
            Results from previously executed nodes
        initial_input : Any
            Initial input data for the pipeline

        Returns
        -------
        Any
            Prepared input data for the node:
            - initial_input if no dependencies
            - dependency result if single dependency
            - dict of dependency results if multiple dependencies

        Examples
        --------
        >>> coordinator = ExecutionCoordinator()
        >>>
        >>> # No dependencies - gets initial input
        >>> # start_input = coordinator.prepare_node_input(
        >>> #     NodeSpec("start", lambda x: x.upper()),
        >>> #     node_results={},
        >>> #     initial_input="hello"
        >>> # )
        >>> # start_input == "hello"
        >>>
        >>> # Single dependency - gets that result directly
        >>> # process_input = coordinator.prepare_node_input(
        >>> #     NodeSpec("process", lambda x: x + "!", deps={"start"}),
        >>> #     node_results={"start": "HELLO"},
        >>> #     initial_input="hello"
        >>> # )
        >>> # process_input == "HELLO"

        Notes
        -----
        The multi-dependency dict preserves node names as keys, making it clear
        where each piece of data came from. This is especially useful for
        debugging and for nodes that need to treat different dependencies
        differently.
        """
        if not node_spec.deps:
            return initial_input

        if len(node_spec.deps) == 1:
            dep_name = next(iter(node_spec.deps))
            return node_results.get(dep_name, initial_input)

        # Multiple dependencies - preserve namespace structure
        aggregated_data = {}
        for dep_name in node_spec.deps:
            if dep_name in node_results:
                aggregated_data[dep_name] = node_results[dep_name]

        return aggregated_data
