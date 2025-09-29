"""Component discovery and registration hooks.

This module provides utilities for discovering components in modules
and registering them during bootstrap.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from types import ModuleType  # noqa: TC003
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexai.core.registry.models import ComponentType
    from hexai.core.registry.registry import ComponentRegistry as RegistryProtocol
else:
    # Runtime imports - needed for actual execution
    from collections.abc import Callable  # noqa: TC003

    from hexai.core.registry.models import ComponentType  # noqa: TC001


def has_hexdag_attrs(obj: object) -> bool:
    """Check if object has hexdag attributes."""
    return hasattr(obj, "_hexdag_type") and hasattr(obj, "_hexdag_name")


logger = logging.getLogger(__name__)


def discover_components(module: ModuleType) -> list[tuple[str, type | Callable | object]]:
    """Discover all components with _hexdag_type in a module.

    Parameters
    ----------
    module : ModuleType
        Module object to scan for components.

    Returns
    -------
    list[tuple[str, type | Callable | object]]
        List of (name, component) tuples for all discovered components.

    Raises
    ------
    TypeError
        If ``module`` is not a :class:`types.ModuleType` instance.
    """

    if not isinstance(module, ModuleType):
        raise TypeError(
            f"Expected ModuleType, got {type(module).__name__}. "
            "Pass a module object, not a string or path."
        )

    components: list[tuple[str, type | Callable | object]] = []

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        # Check if object has our hexdag type marker and is a class or callable
        if hasattr(obj, "_hexdag_type") and (inspect.isclass(obj) or callable(obj)):
            # Check if it's from this module or a submodule
            obj_module = getattr(obj, "__module__", None)
            if obj_module and (
                obj_module == module.__name__ or obj_module.startswith(module.__name__ + ".")
            ):
                components.append((name, obj))

    return components


def _get_component_names(component: object) -> list[str]:
    """Get all names a component should be registered under (including aliases).

    Parameters
    ----------
    component : object
        The component object

    Returns
    -------
    list[str]
        List of all names to register under
    """
    if hasattr(component, "_hexdag_names"):
        names = getattr(component, "_hexdag_names")  # noqa: B009
        return names if isinstance(names, list) else [names]
    if hasattr(component, "_hexdag_name"):
        name = getattr(component, "_hexdag_name")  # noqa: B009
        return [name] if isinstance(name, str) else []
    return []


def _register_component_direct(
    registry: RegistryProtocol,
    component: object,
    namespace: str,
    module_path: str,
) -> int:
    """Register a component directly from its attributes.

    Parameters
    ----------
    registry : RegistryProtocol
        The registry to register into
    component : object
        The component to register
    namespace : str
        Namespace to register under
    module_path : str
        Source module path for logging

    Returns
    -------
    int
        Number of successful registrations
    """
    # Extract attributes directly from component
    component_type = component._hexdag_type  # type: ignore[attr-defined]
    names_to_register = _get_component_names(component)
    subtype = getattr(component, "_hexdag_subtype", None)
    description = getattr(component, "_hexdag_description", "")

    privileged = namespace == "core"
    count = 0

    for name in names_to_register:
        try:
            registry.register(
                name=name,
                component=component,
                component_type=component_type,
                namespace=namespace,
                privileged=privileged,
                subtype=subtype,
                description=description,
            )
            count += 1

            # Log appropriate message based on type
            if component_type == ComponentType.PORT:
                logger.debug("Registered port %s:%s from %s", namespace, name, module_path)
            else:
                component_type_str = (
                    component_type if isinstance(component_type, str) else "component"
                )
                logger.debug(
                    "Registered %s %s:%s from %s",
                    component_type_str,
                    namespace,
                    name,
                    module_path,
                )
        except Exception as e:
            component_type_str = (
                "port" if component_type == ComponentType.PORT else str(component_type)
            )
            logger.error(
                f"Failed to register {component_type_str} {namespace}:{name} from {module_path}"
                f": {e}"
            )
            raise

    return count


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

    seen_objects = set()
    unique_components = []
    for name, component in components:
        if id(component) not in seen_objects:
            seen_objects.add(id(component))
            unique_components.append((name, component))
    components = unique_components

    logger.debug("Phase A: Registering ports from %s", module_path)

    for _, component in components:
        if not hasattr(component, "_hexdag_type"):
            continue

        component_type = getattr(component, "_hexdag_type")  # noqa: B009
        if component_type == ComponentType.PORT:
            count += _register_component_direct(registry, component, namespace, module_path)

    # Phase B: Register everything else
    logger.debug("Phase B: Registering adapters and other components from %s", module_path)
    for _, component in components:
        if not hasattr(component, "_hexdag_type"):
            continue

        component_type = getattr(component, "_hexdag_type")  # noqa: B009
        if component_type != ComponentType.PORT:
            count += _register_component_direct(registry, component, namespace, module_path)

    return count
