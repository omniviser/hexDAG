"""Backwards compatibility — HexDAGLib is now Service.

.. deprecated::
    ``HexDAGLib`` is deprecated.  Use :class:`~hexdag.kernel.service.Service`
    with ``@tool`` / ``@step`` decorators instead.

This module re-exports :class:`Service` as ``HexDAGLib`` so that existing
imports (``from hexdag.stdlib.lib_base import HexDAGLib``) continue to work
during the migration period.  ``get_lib_tool_schemas`` is aliased to
``get_service_tool_schemas``.
"""

from hexdag.kernel.service import Service as HexDAGLib
from hexdag.kernel.service import get_service_tool_schemas as get_lib_tool_schemas

__all__ = ["HexDAGLib", "get_lib_tool_schemas"]
