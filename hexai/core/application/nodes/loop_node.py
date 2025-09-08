"""LoopNode for creating loop control nodes with conditional execution.

This module provides the LoopNode factory that creates nodes capable of:
- Loop iteration control
- Conditional execution based on success criteria
- State management across iterations
- Dynamic routing decisions
"""

import logging
from typing import Any, Callable

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from .base_node_factory import BaseNodeFactory


@node(name="loop_node", subtype=NodeSubtype.LOOP, namespace="core")
class LoopNode(BaseNodeFactory):
    """Factory class for creating loop control nodes with iteration management."""

    def __init__(self) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

    def __call__(
        self,
        name: str,
        max_iterations: int = 3,
        success_condition: Callable[[Any], bool] | None = None,
        iteration_key: str = "loop_iteration",
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a loop control node.

        Args
        ----
            name: Name of the node
            max_iterations: Maximum number of loop iterations
            success_condition: Function to evaluate if success criteria are met
            iteration_key: Key to store iteration count in state
            **kwargs: Additional arguments for NodeSpec

        Returns
        -------
            NodeSpec configured for loop control
        """
        # Validate max_iterations
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        async def loop_controller_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Execute loop control logic."""
            logger = logging.getLogger("hexai.app.application.nodes.loop_node")
            logger.info("🔄 LOOP NODE: %s", name)

            # Get current iteration count from input state
            if isinstance(input_data, dict):
                current_iteration = input_data.get(iteration_key, 0) + 1
                loop_history = input_data.get(f"{name}_history", [])
                state = input_data
            else:
                current_iteration = 1
                loop_history = []
                state = {"input": input_data}

            logger.info("🔢 Starting iteration %d/%d", current_iteration, max_iterations)

            # Extract relevant data for evaluation
            if isinstance(input_data, dict):
                data_dict = input_data
            else:
                # Try to get model_dump if available, otherwise use as-is
                try:
                    data_dict = input_data.model_dump()
                except (AttributeError, TypeError):
                    data_dict = {"input": input_data}

            # Evaluate success condition
            success_criteria_met = False
            if success_condition is not None:
                try:
                    success_criteria_met = success_condition(data_dict)
                    logger.debug("🎯 Success condition evaluated: %s", success_criteria_met)
                except Exception as e:
                    logger.error("❌ Success condition evaluation failed: %s", e)
                    success_criteria_met = False
            else:
                # Default success condition: check for termination signals
                response_text = str(data_dict.get("response", ""))
                success_criteria_met = "Tool_END" in response_text or "PHASE" in response_text
                logger.debug(
                    "🎯 Default success criteria: response contains termination -> %s",
                    success_criteria_met,
                )

            # Determine if we should continue looping
            should_continue = current_iteration < max_iterations and not success_criteria_met

            # Update state
            state[iteration_key] = current_iteration
            state[f"{name}_history"] = loop_history + [data_dict]
            state["should_continue"] = should_continue
            state["success_criteria_met"] = success_criteria_met
            state["success"] = success_criteria_met  # Add for test compatibility
            state["iterations_completed"] = current_iteration

            logger.info(
                "🎪 Loop control result: iteration=%d, continue=%s, success=%s",
                current_iteration,
                should_continue,
                success_criteria_met,
            )

            return state

        return NodeSpec(
            name=name,
            fn=loop_controller_fn,
            **kwargs,
        )


@node(name="conditional_node", subtype=NodeSubtype.CONDITIONAL, namespace="core")
class ConditionalNode(BaseNodeFactory):
    """Factory class for creating conditional routing nodes."""

    def __init__(self) -> None:
        """Initialize ConditionalNode factory."""
        super().__init__()

    def __call__(
        self,
        name: str,
        condition_key: str = "should_continue",
        true_action: str = "continue",
        false_action: str = "proceed",
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a conditional routing node.

        Args
        ----
            name: Name of the node
            condition_key: Key in input data to evaluate for routing decision
            true_action: Action to take when condition is true
            false_action: Action to take when condition is false
            **kwargs: Additional arguments for NodeSpec

        Returns
        -------
            NodeSpec configured for conditional routing
        """

        async def conditional_router_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Execute conditional routing logic."""
            logger = logging.getLogger("hexai.app.application.nodes.conditional_node")
            logger.info("🧭 CONDITIONAL NODE: %s", name)

            # Extract data
            if hasattr(input_data, "model_dump"):
                data_dict = input_data.model_dump()
            elif isinstance(input_data, dict):
                data_dict = input_data
            else:
                data_dict = {"input": input_data}

            # Evaluate condition
            condition_raw = data_dict.get(condition_key, False)
            condition_value = bool(condition_raw)  # Convert to boolean
            action = true_action if condition_value else false_action

            logger.info(
                "🚦 Condition '%s' = %s -> action: %s", condition_key, condition_value, action
            )

            # Prepare routing result
            result = {
                "condition_key": condition_key,
                "condition_value": condition_value,
                "action": action,
                "routing_decision": (
                    f"{'continue' if condition_value else 'proceed'} based on {condition_key}"
                ),
            }

            # Preserve original input data
            result.update(data_dict)

            logger.debug("✅ Conditional routing complete")
            return result

        return NodeSpec(
            name=name,
            fn=conditional_router_fn,
            **kwargs,
        )
