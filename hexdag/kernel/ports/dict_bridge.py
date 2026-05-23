"""Dict↔model bridge utilities for port methods.

Ports that define typed methods with Pydantic request/response models often
need a dict-based companion for pipeline/node compatibility.  These utilities
eliminate the boilerplate of manual field extraction and serialization.

Example — providing a default dict method on a protocol::

    from hexdag.kernel.ports.dict_bridge import dict_bridge_call

    class SupportsEmail(Protocol):
        async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
            ...

        async def send_email_from_dict(self, request: dict) -> dict:
            return await dict_bridge_call(
                self.send_email, request, SendEmailRequest,
            )

Adapters that implement ``send_email`` automatically get
``send_email_from_dict`` for free — no copy-paste needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from pydantic import BaseModel


async def dict_bridge_call(
    method: Callable[..., Coroutine[Any, Any, Any]],
    request: dict[str, Any],
    request_model: type[BaseModel],
) -> dict[str, Any]:
    """Call a typed async port method with a plain dict, return a plain dict.

    1. Validates *request* into *request_model* via ``model_validate``.
    2. Awaits *method* with the typed request.
    3. Serialises the result back to a dict via ``model_dump(mode="json")``.
       If the result is already a dict it is returned as-is.

    Parameters
    ----------
    method:
        The typed async method to call (e.g. ``self.send_email``).
    request:
        Raw dict payload from the pipeline/node layer.
    request_model:
        Pydantic model class for the request (e.g. ``SendEmailRequest``).

    Returns
    -------
    dict[str, Any]
        JSON-safe dict representation of the result.
    """
    typed_request = request_model.model_validate(request)
    result = await method(typed_request)
    if hasattr(result, "model_dump"):
        return dict(result.model_dump(mode="json"))
    if isinstance(result, dict):
        return result
    return {"result": result}
