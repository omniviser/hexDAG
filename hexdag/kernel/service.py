"""Service base class — unified abstraction for port-backed operations.

A **Service** wraps one or more ports/adapters behind a stable API.
Methods are explicitly annotated with :func:`tool` and/or :func:`step`
decorators to declare how they can be invoked:

- ``@tool`` — available as an agent-callable tool during ReAct reasoning
- ``@step`` — available as a deterministic DAG node
- Both decorators can be stacked on the same method.

Lifecycle
---------
1. The orchestrator instantiates the service and calls :meth:`asetup`.
2. During pipeline execution:
   - ``@step`` methods may be invoked as DAG nodes
   - ``@tool`` methods may be called by agent nodes
3. After the pipeline finishes, the orchestrator calls :meth:`ateardown`.

Creating a custom service
-------------------------
.. code-block:: python

    from hexdag.kernel.service import Service, tool, step

    class OrderService(Service):
        def __init__(self, store: SupportsKeyValue) -> None:
            self._store = store

        @tool
        async def get_order(self, order_id: str) -> dict:
            \"\"\"Get order by ID.\"\"\"
            return await self._store.aget(f"order:{order_id}")

        @step
        async def save_order(self, order_id: str, data: dict) -> dict:
            \"\"\"Persist an order.\"\"\"
            await self._store.aset(f"order:{order_id}", data)
            return {"saved": True, "order_id": order_id}

        @tool
        @step
        async def validate_order(self, order_id: str) -> dict:
            \"\"\"Validate order data — usable as both tool and step.\"\"\"
            ...

YAML configuration::

    spec:
      services:
        orders:
          class: myapp.services.OrderService
          config:
            store: database
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Decorator markers
# ---------------------------------------------------------------------------

_TOOL_ATTR = "_hexdag_tool"
_STEP_ATTR = "_hexdag_step"


def tool[F: Callable[..., Any]](fn: F) -> F:
    """Mark an async method as an agent-callable tool.

    Agent nodes can invoke ``@tool`` methods during their ReAct reasoning
    loop.  The method's signature and docstring are used to generate the
    tool schema automatically.

    Example
    -------
    .. code-block:: python

        class MyService(Service):
            @tool
            async def search(self, query: str) -> list[dict]:
                \"\"\"Search the database.\"\"\"
                ...
    """
    setattr(fn, _TOOL_ATTR, True)
    return fn


def step[F: Callable[..., Any]](fn: F) -> F:
    """Mark an async method as a deterministic DAG step.

    ``@step`` methods can be invoked as nodes in a pipeline via the
    ``service_call`` node kind.  They receive structured input and
    produce structured output, just like any other DAG node.

    Example
    -------
    .. code-block:: python

        class MyService(Service):
            @step
            async def transform(self, data: dict) -> dict:
                \"\"\"Transform input data.\"\"\"
                ...
    """
    setattr(fn, _STEP_ATTR, True)
    return fn


def _is_tool(method: Any) -> bool:
    """Return True if *method* is decorated with ``@tool``."""
    return getattr(method, _TOOL_ATTR, False) is True


def _is_step(method: Any) -> bool:
    """Return True if *method* is decorated with ``@step``."""
    return getattr(method, _STEP_ATTR, False) is True


# ---------------------------------------------------------------------------
# Service base class
# ---------------------------------------------------------------------------


class Service:
    """Base class for hexDAG services.

    A service wraps one or more ports/adapters behind a stable, typed API.
    Public methods decorated with :func:`tool` are exposed to agent nodes;
    methods decorated with :func:`step` can be used as deterministic DAG
    nodes.

    Subclasses receive ports via constructor injection following the
    explicit-parameters convention (no ``**kwargs``).
    """

    async def asetup(self) -> None:
        """Called once before the pipeline starts.

        Override to perform one-time initialisation (schema creation,
        connection warming, etc.).
        """

    async def ateardown(self) -> None:
        """Called once after the pipeline finishes.

        Override to release resources, flush buffers, etc.
        """

    def get_tools(self) -> dict[str, Callable[..., Any]]:
        """Return ``{name: bound_method}`` for all ``@tool``-decorated methods."""
        return self._collect_methods(_is_tool)

    def get_steps(self) -> dict[str, Callable[..., Any]]:
        """Return ``{name: bound_method}`` for all ``@step``-decorated methods."""
        return self._collect_methods(_is_step)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_methods(self, predicate: Callable[[Any], bool]) -> dict[str, Callable[..., Any]]:
        """Collect bound methods matching *predicate*."""
        methods: dict[str, Callable[..., Any]] = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name, None)
            if attr is None:
                continue
            if callable(attr) and predicate(attr):
                methods[name] = attr
        return methods

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        tools = list(self.get_tools().keys())
        steps = list(self.get_steps().keys())
        return f"{type(self).__name__}(tools={tools}, steps={steps})"


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_STR_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if isinstance(annotation, str):
        return {"type": _STR_TYPE_MAP.get(annotation, "string")}
    return {"type": _TYPE_MAP.get(annotation, "string")}


def get_service_tool_schemas(service: Service) -> list[dict[str, Any]]:
    """Generate OpenAI-compatible tool schemas from a service's ``@tool`` methods.

    Parameters
    ----------
    service : Service
        The service instance to introspect.

    Returns
    -------
    list[dict[str, Any]]
        List of tool schema dicts suitable for LLM function-calling APIs.
    """
    schemas: list[dict[str, Any]] = []
    for name, method in service.get_tools().items():
        sig = inspect.signature(method)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            prop: dict[str, Any] = {"type": "string"}
            if param.annotation is not inspect.Parameter.empty:
                prop = _annotation_to_json_schema(param.annotation)
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": (method.__doc__ or "").strip().split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        schemas.append(schema)
    return schemas


def get_service_step_schemas(service: Service) -> list[dict[str, Any]]:
    """Generate schemas from a service's ``@step`` methods.

    Same format as :func:`get_service_tool_schemas` but for step methods.
    """
    schemas: list[dict[str, Any]] = []
    for name, method in service.get_steps().items():
        sig = inspect.signature(method)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            prop: dict[str, Any] = {"type": "string"}
            if param.annotation is not inspect.Parameter.empty:
                prop = _annotation_to_json_schema(param.annotation)
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": (method.__doc__ or "").strip().split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        schemas.append(schema)
    return schemas
