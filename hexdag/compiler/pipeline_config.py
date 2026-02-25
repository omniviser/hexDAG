"""Backwards-compatibility shim — models moved to ``hexdag.kernel.domain.pipeline_config``.

.. deprecated::
    Import from ``hexdag.kernel.domain.pipeline_config`` instead.
"""

from hexdag.kernel.domain.pipeline_config import (
    BaseNodeConfig,
    CustomTypeConfig,
    PipelineConfig,
)

__all__ = ["BaseNodeConfig", "CustomTypeConfig", "PipelineConfig"]
