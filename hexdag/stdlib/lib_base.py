"""Base class for system libraries (libs).

Libs are the hexDAG equivalent of Linux shared libraries (``libc``, ``libm``).
Every public async method whose name starts with ``a`` (e.g. ``aquery``,
``acreate_order``) is auto-exposed as an agent-callable tool.

Lifecycle
---------
1. The orchestrator instantiates the lib and calls :meth:`asetup`.
2. During pipeline execution, agent nodes can invoke any tool returned by
   :meth:`get_tools`.
3. After the pipeline finishes, the orchestrator calls :meth:`ateardown`.

Creating a custom lib
---------------------
.. code-block:: python

    class OrderManager(HexDAGLib):
        def __init__(self, store: SupportsKeyValue) -> None:
            self._store = store

        async def acreate_order(self, customer_id: str, items: list[dict]) -> str:
            order_id = str(uuid4())
            await self._store.aset(f"order:{order_id}", {"customer": customer_id, "items": items})
            return order_id

YAML configuration::

    spec:
      libs:
        orders:
          class: myapp.lib.OrderManager
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class HexDAGLib:
    """Base class for system libraries.

    Subclasses receive ports via constructor injection and expose
    public async ``a*`` methods as agent-callable tools.
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
        """Return a mapping of tool-name → async callable.

        By default every public async method whose name starts with ``a``
        (but not ``_``) is included.  Override to customise.
        """
        tools: dict[str, Callable[..., Any]] = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            if not name.startswith("a"):
                continue
            # Skip lifecycle methods
            if name in ("asetup", "ateardown"):
                continue
            attr = getattr(self, name)
            if callable(attr) and asyncio.iscoroutinefunction(attr):
                tools[name] = attr
        return tools

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        tool_names = list(self.get_tools().keys())
        return f"{type(self).__name__}(tools={tool_names})"


def get_lib_tool_schemas(lib: HexDAGLib) -> list[dict[str, Any]]:
    """Generate tool schemas from a lib's exposed tools.

    Returns a list of OpenAI-compatible tool schema dicts, one per tool.
    Each schema is derived from the method's signature and docstring.
    """
    schemas: list[dict[str, Any]] = []
    for name, method in lib.get_tools().items():
        sig = inspect.signature(method)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            prop: dict[str, Any] = {"type": "string"}
            # Infer type from annotation
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
    # Handle string annotations (from __future__ import annotations)
    if isinstance(annotation, str):
        return {"type": _STR_TYPE_MAP.get(annotation, "string")}
    # Handle actual types
    return {"type": _TYPE_MAP.get(annotation, "string")}
