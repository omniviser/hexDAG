"""TransitionNode — built-in node for entity state transitions.

Validates and executes a state transition through the pipeline-scoped
``EntityState`` service.  Works in both workflow (DAG step) and agentic
(agent tool call) modes — both paths go through ``EntityState.atransition()``.

YAML usage::

    - kind: transition
      metadata:
        name: accept_load
      spec:
        entity: load
        entity_id: $input.load_id
        to_state: ACCEPTED
        reason: "Smart acceptance criteria met"

Dynamic references are auto-wired via ``input_mapping`` so the
ExecutionCoordinator resolves ``node.field`` / ``$input.field``
paths before the node function runs::

    - kind: transition
      metadata:
        name: escalate
      spec:
        entity: load
        entity_id: resolved_match.load_id          # resolved at runtime
        to_state: ESCALATED
        reason: build_escalation_reason.reason      # resolved at runtime
"""

from __future__ import annotations

import re
from typing import Any

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

# Patterns that indicate a value needs runtime resolution via input_mapping
# rather than being passed as a literal string.
_DYNAMIC_RE = re.compile(
    r"^(\$input\.|[A-Za-z_][A-Za-z0-9_]*\.)"  # $input.field or node.field
)


def _is_dynamic_ref(value: str | None) -> bool:
    """Return True if *value* looks like a runtime reference."""
    return value is not None and bool(_DYNAMIC_RE.match(value))


class TransitionNode(BaseNodeFactory, yaml_alias="transition"):
    """Factory for entity state transition nodes.

    Creates a node that transitions a business entity to a new state
    via the pipeline-scoped ``EntityState`` service.  The transition is
    validated against the registered state machine, fires transition
    handlers, and emits ``StateTransitionEvent``.
    """

    __aliases__ = ("transition_node",)

    def __call__(
        self,
        name: str,
        *,
        entity: str,
        entity_id: str | None = None,
        to_state: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Build a transition node spec.

        Parameters
        ----------
        name : str
            Node name in the DAG.
        entity : str
            Entity type (must match a registered state machine).
        entity_id : str | None
            Entity ID expression — resolved at runtime from input.
            Supports ``$input.field`` and ``node_name.field`` syntax.
        to_state : str
            Target state for the transition.
        reason : str | None
            Optional reason expression — resolved at runtime.
            Supports ``$input.field`` and ``node_name.field`` syntax.
        **kwargs
            Additional NodeSpec fields (when, on_error, etc.).
        """
        _entity = entity
        _to_state = to_state

        # Build input_mapping so the ExecutionCoordinator resolves dynamic
        # references (node.field, $input.field) before the node runs.
        input_mapping: dict[str, str] = {}
        if _is_dynamic_ref(entity_id):
            input_mapping["entity_id"] = entity_id  # type: ignore[assignment]
        if _is_dynamic_ref(reason):
            input_mapping["reason"] = reason  # type: ignore[assignment]

        # Keep raw values for the closure fallback (literal strings / None)
        _entity_id_literal = entity_id if not _is_dynamic_ref(entity_id) else None
        _reason_literal = reason if not _is_dynamic_ref(reason) else None

        async def _transition(inputs: dict[str, Any], **kw: Any) -> dict[str, Any]:
            from hexdag.kernel.context.execution_context import (  # lazy: runtime-only closure
                get_current_node_name,
                get_pipeline_name,
                get_run_id,
                get_services,
            )
            from hexdag.kernel.orchestration.events.events import (  # lazy: runtime-only closure
                TransitionContext,
            )

            services = get_services()
            if not services or "entity_state" not in services:
                msg = (
                    f"TransitionNode '{name}' requires an 'entity_state' service. "
                    "Declare spec.state_machines or register EntityState as a service."
                )
                raise RuntimeError(msg)

            entity_state = services["entity_state"]

            # Resolve entity_id: input_mapping-resolved value first, then literal.
            # Use .get() with default to preserve falsy resolved values (0, "", False).
            resolved_id = inputs.get("entity_id", _entity_id_literal)

            if resolved_id is None:
                msg = f"TransitionNode '{name}': entity_id could not be resolved"
                raise ValueError(msg)

            # Resolve reason: input_mapping-resolved value first, then literal.
            resolved_reason = inputs.get("reason", _reason_literal)

            context = TransitionContext(
                run_id=get_run_id() or "",
                pipeline_name=get_pipeline_name() or "",
                node_name=get_current_node_name() or name,
            )

            result: dict[str, Any] = await entity_state.atransition(
                entity_type=_entity,
                entity_id=str(resolved_id),
                to_state=_to_state,
                reason=str(resolved_reason) if resolved_reason else None,
                _context=context,
            )
            return result

        # Extract NodeSpec-level kwargs
        when = kwargs.pop("when", None)
        on_error = kwargs.pop("on_error", None)

        return NodeSpec(
            name=name,
            fn=_transition,
            params={"input_mapping": input_mapping} if input_mapping else {},
            when=when,
            on_error=on_error,
            factory_class="hexdag.stdlib.nodes.transition_node.TransitionNode",
            factory_params={
                "entity": _entity,
                "entity_id": entity_id,
                "to_state": _to_state,
                "reason": reason,
            },
        )
