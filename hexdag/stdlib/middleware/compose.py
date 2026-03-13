"""Middleware composition utility."""

from typing import Any


def compose(adapter: Any, *middleware_classes: type) -> Any:
    """Build a middleware chain: adapter → layer1 → layer2 → ...

    Each middleware class must accept the inner port as its first argument.
    Layers are applied left-to-right (first class wraps the adapter directly).

    Parameters
    ----------
    adapter : Any
        The base port adapter
    *middleware_classes : type
        Middleware classes to stack, in inner-to-outer order

    Returns
    -------
    Any
        The outermost middleware wrapping the adapter
    """
    result = adapter
    for cls in middleware_classes:
        result = cls(result)
    return result
