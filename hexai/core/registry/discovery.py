"""Component discovery and registration hooks.

This module provides utilities for discovering components in modules
and registering them during bootstrap.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from types import ModuleType  # noqa: TC003
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexai.core.registry.models import DecoratorMetadata
    from hexai.core.registry.registry import ComponentRegistry as RegistryProtocol
else:
    # Runtime imports - needed for actual execution
    from collections.abc import Callable  # noqa: TC003

    from hexai.core.registry.models import DecoratorMetadata  # noqa: TC001


class ComponentWithMetadata(Protocol):
    """Protocol for components with HexDAG metadata."""

    __hexdag_metadata__: DecoratorMetadata


logger = logging.getLogger(__name__)


def discover_components(module: ModuleType) -> list[tuple[str, type | Callable | object]]:
    """Discover all components with __hexdag_metadata__ in a module.

    Parameters
    ----------
    module : ModuleType
        The module to scan for components.

    Returns
    -------
    list[tuple[str, type | Callable | object]]
        List of (name, component) tuples for all discovered components.

    Raises
    ------
    TypeError
        If module is not a ModuleType instance.
    """
    if not isinstance(module, ModuleType):
        raise TypeError(
            f"Expected ModuleType, got {type(module).__name__}. "
            "Pass a module object, not a string or path."
        )

    components = []

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        # Check if object has our metadata marker and is a class or callable
        if hasattr(obj, "__hexdag_metadata__") and (inspect.isclass(obj) or callable(obj)):
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
        logger.error("Failed to import module '%s': %s", module_path, e)
        raise

    # Check if module has a custom register_components function
    if hasattr(module, "register_components"):
        # Module has custom registration logic
        logger.debug("Using custom register_components from %s", module_path)
        result = module.register_components(registry, namespace)
        # Ensure we return an int
        return int(result) if result is not None else 0

    # Use standard discovery
    components = discover_components(module)
    count = 0

    # Two-phase registration: Phase A (ports first)
    logger.debug("Phase A: Registering ports from %s", module_path)
    ports_registered = []

    for _, component in components:
        # Type guard - we know from discover_components that these have metadata
        if not hasattr(component, "__hexdag_metadata__"):
            continue
        # Use getattr to access dynamic attribute (type checkers can't verify this)
        metadata: DecoratorMetadata = getattr(component, "__hexdag_metadata__")  # noqa: B009

        # Skip non-ports in phase A
        if metadata.type != "PORT":
            continue

        meta_name = metadata.name
        meta_type = metadata.type
        meta_subtype = metadata.subtype
        meta_description = metadata.description
        meta_adapter = metadata.adapter_metadata

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
                adapter_metadata=meta_adapter,
            )
            count += 1
            ports_registered.append(meta_name)
            logger.debug("Registered port %s:%s from %s", actual_namespace, meta_name, module_path)
        except Exception as e:
            logger.error(
                f"Failed to register port {actual_namespace}:{meta_name} from {module_path}: {e}"
            )
            raise

    # Phase B: Register adapters (and everything else)
    logger.debug("Phase B: Registering adapters and other components from %s", module_path)

    for _, component in components:
        # Type guard - we know from discover_components that these have metadata
        if not hasattr(component, "__hexdag_metadata__"):
            continue
        # Use getattr to access dynamic attribute (type checkers can't verify this)
        component_metadata: DecoratorMetadata = getattr(component, "__hexdag_metadata__")  # noqa: B009

        # Skip ports (already registered in phase A)
        if component_metadata.type == "PORT":
            continue

        meta_name = component_metadata.name
        meta_type = component_metadata.type
        meta_subtype = component_metadata.subtype
        meta_description = component_metadata.description
        meta_adapter = component_metadata.adapter_metadata

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
                adapter_metadata=meta_adapter,
            )
            count += 1
            component_type_str = meta_type if isinstance(meta_type, str) else "component"
            logger.debug(
                "Registered %s %s:%s from %s",
                component_type_str,
                actual_namespace,
                meta_name,
                module_path,
            )
        except Exception as e:
            logger.error(
                f"Failed to register {actual_namespace}:{meta_name} from {module_path}: {e}"
            )
            raise

    return count
