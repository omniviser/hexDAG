"""Adapter implementations for external services.

When `import hexai.adapters` is called, all adapter modules
in this package will be imported automatically,
triggering their @register_port decorators.
"""

import importlib
import pkgutil
import sys
from types import ModuleType

from .function_tool_router import FunctionBasedToolRouter
from .in_memory_memory import InMemoryMemory

for module_info in pkgutil.iter_modules(__path__, __name__ + "."):
    if module_info.name not in [
        "hexai.adapters.in_memory_memory",
        "hexai.adapters.function_tool_router",
    ]:
        module: ModuleType = importlib.import_module(module_info.name)
        for attr in getattr(module, "__all__", []):
            setattr(sys.modules[__name__], attr, getattr(module, attr))

__all__ = ["InMemoryMemory", "FunctionBasedToolRouter"]
