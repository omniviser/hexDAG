"""ServiceCallNode — call a @step method on a Service as a DAG node.

This node provides direct YAML-to-Service wiring: YAML pipelines reference
a service name and method, and at execution time the bound ``@step`` method
is called with the node's input.  No intermediate bridge layer.

YAML usage::

    - kind: service_call_node          # or system:service_call
      metadata:
        name: save_order
      spec:
        service: orders                # references spec.services.orders
        method: save_order             # @step method name
        input_mapping:
          order_id: "extract_order.order_id"
          data: "extract_order.details"
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from hexdag.kernel.context import get_services
from hexdag.kernel.logging import get_logger
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from hexdag.kernel.domain.dag import NodeSpec

logger = get_logger(__name__)


class ServiceCallOutput(BaseModel):
    """Output envelope from a service call."""

    result: Any
    service: str
    method: str
    error: str | None = None


class ServiceCallNode(BaseNodeFactory, yaml_alias="service_call_node"):
    """Call a ``@step`` method on a Service as a deterministic DAG node.

    At execution time, the node:

    1. Retrieves services from the execution context.
    2. Looks up the named service.
    3. Looks up the method via ``service.get_steps()`` (validates ``@step``).
    4. Calls the bound method with the node's input data.
    5. Returns a :class:`ServiceCallOutput` envelope.

    Examples
    --------
    Direct (programmatic) usage::

        factory = ServiceCallNode()
        node = factory(
            name="save_order",
            service="orders",
            method="save_order",
        )

    YAML usage::

        - kind: service_call_node
          metadata:
            name: save_order
          spec:
            service: orders
            method: save_order
    """

    # Studio UI metadata
    _hexdag_icon: ClassVar[str] = "Cog"
    _hexdag_color: ClassVar[str] = "#10b981"  # emerald-500

    # System kind marker — enables ``system:`` namespace aliases in discovery.
    _hexdag_system_kind: ClassVar[bool] = True

    def __call__(
        self,
        name: str,
        service: str,
        method: str,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec that calls a ``@step`` method on a Service.

        Parameters
        ----------
        name:
            Node name (must be unique within the pipeline).
        service:
            Name of the service declared in ``spec.services``.
        method:
            Name of the ``@step`` method to call.
        deps:
            Explicit dependencies.
        **kwargs:
            Framework params (``timeout``, ``max_retries``, ``when``, etc.)
            and any extra params forwarded to ``create_node_with_mapping``.
        """
        service_name = service
        method_name = method

        async def execute_service_call(input_data: dict[str, Any]) -> dict[str, Any]:
            """Resolve service + method at runtime and call with *input_data*."""
            # 1. Get services from execution context
            services = get_services()
            if not services:
                return {
                    "result": None,
                    "service": service_name,
                    "method": method_name,
                    "error": "No services available in execution context",
                }

            # 2. Look up service instance
            svc = services.get(service_name)
            if svc is None:
                available = list(services.keys())
                return {
                    "result": None,
                    "service": service_name,
                    "method": method_name,
                    "error": (f"Service '{service_name}' not found. Available: {available}"),
                }

            # 3. Look up method via get_steps() — validates @step
            steps = svc.get_steps()
            step_fn = steps.get(method_name)
            if step_fn is None:
                available_steps = list(steps.keys())
                return {
                    "result": None,
                    "service": service_name,
                    "method": method_name,
                    "error": (
                        f"Method '{method_name}' is not a @step on "
                        f"service '{service_name}'. "
                        f"Available steps: {available_steps}"
                    ),
                }

            # 4. Call the bound method
            with node_timer() as t:
                try:
                    if inspect.iscoroutinefunction(step_fn):
                        result = await step_fn(**input_data)
                    else:
                        result = step_fn(**input_data)

                    logger.debug(
                        "Service step {}.{} completed in {}ms",
                        service_name,
                        method_name,
                        t.duration_str,
                    )

                    return {
                        "result": result,
                        "service": service_name,
                        "method": method_name,
                        "error": None,
                    }
                except Exception as e:
                    logger.warning(
                        "Service step {}.{} failed after {}ms: {}",
                        service_name,
                        method_name,
                        t.duration_str,
                        e,
                    )
                    return {
                        "result": None,
                        "service": service_name,
                        "method": method_name,
                        "error": str(e),
                    }

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=execute_service_call,
            input_schema=None,
            output_schema=ServiceCallOutput,
            deps=deps,
            **kwargs,
        )
