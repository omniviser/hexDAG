"""Component discovery and registration hooks.

This module provides utilities for discovering components in modules
and registering them during bootstrap.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any

from hexai.core.registry.models import DecoratorMetadata

if TYPE_CHECKING:
    from hexai.core.registry.registry import ComponentRegistry as RegistryProtocol
else:
    RegistryProtocol = Any

logger = logging.getLogger(__name__)


def discover_components(module: Any) -> list[tuple[str, Any]]:
    """Discover all components with __hexdag_metadata__ in a module.

    Parameters
    ----------
    module : Any
        The module to scan for components.

    Returns
    -------
    list[tuple[str, Any]]
        List of (name, component) tuples for all discovered components.
    """
    components = []

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        # Check if object has our metadata marker
        if hasattr(obj, "__hexdag_metadata__"):
            # Only register classes and callables defined in this module
            if inspect.isclass(obj) or callable(obj):
                # Check if it's from this module or a submodule
                obj_module = getattr(obj, "__module__", None)
                if obj_module and (
                    obj_module == module.__name__ or obj_module.startswith(module.__name__ + ".")
                ):
                    components.append((name, obj))

    return components


def register_components(registry: RegistryProtocol, namespace: str, module_path: str) -> int:
    """Register components from a module using standard hook.

    This is the standard registration hook that modules can use or override.
    It discovers all decorated components in the module and registers them.

    Parameters
    ----------
    registry: RegistryProtocol
        The registry to register components into.
    namespace: str
        The namespace to register under (from manifest).
    module_path: str
        The module path to import and scan.

    Returns
    -------
    int
        Number of components registered.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Failed to import module '{module_path}': {e}")
        raise

    # Check if module has a custom register_components function
    if hasattr(module, "register_components"):
        # Module has custom registration logic
        logger.debug(f"Using custom register_components from {module_path}")
        result = module.register_components(registry, namespace)
        # Ensure we return an int
        return int(result) if result is not None else 0

    # Use standard discovery
    components = discover_components(module)
    count = 0

    for _, component in components:
        metadata: DecoratorMetadata = component.__hexdag_metadata__
        meta_name = metadata.name
        meta_type = metadata.type
        meta_subtype = metadata.subtype
        meta_description = metadata.description

        # Namespace from manifest overrides decorator's declared_namespace
        actual_namespace = namespace

        # Determine if this needs privileged access
        privileged = actual_namespace == "core"

        try:
            registry.register(
                name=meta_name,
                component=component,
                component_type=meta_type,
                namespace=actual_namespace,
                privileged=privileged,
                subtype=meta_subtype,
                description=meta_description,
            )
            count += 1
            logger.debug(f"Registered {actual_namespace}:{meta_name} from {module_path}")
        except Exception as e:
            logger.error(
                f"Failed to register {actual_namespace}:{meta_name} from {module_path}: {e}"
            )
            raise

    return count
