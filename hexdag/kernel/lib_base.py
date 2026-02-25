"""Backwards compatibility — HexDAGLib is now Service.

.. deprecated::
    ``HexDAGLib`` is deprecated.  Use :class:`~hexdag.kernel.service.Service`
    with ``@tool`` / ``@step`` decorators instead.
"""

from hexdag.kernel.service import Service as HexDAGLib
from hexdag.kernel.service import get_service_tool_schemas as get_lib_tool_schemas

__all__ = ["HexDAGLib", "get_lib_tool_schemas"]
