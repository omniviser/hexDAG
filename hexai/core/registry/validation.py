"""Registry validation and attribute extraction.

Provides RegistryValidator class that combines validation logic and
attribute extraction for component registration.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from hexai.core.registry.exceptions import InvalidComponentError
from hexai.core.registry.models import (
    COMPONENT_VALUE_ATTR,
    IMPLEMENTS_PORT_ATTR,
    REQUIRED_PORTS_ATTR,
    ClassComponent,
    ComponentType,
    FunctionComponent,
    InstanceComponent,
    Namespace,
)


class RegistryValidator:
    """Validates components and extracts metadata for registry operations.

    This class provides:
    - Component type, name, and namespace validation
    - Component wrapping (class/function/instance)
    - Attribute extraction (ports, requirements)
    - Namespace protection checking
    """

    # Protected namespaces that require privileged registration
    PROTECTED_NAMESPACES = {Namespace.CORE}

    # Attribute extraction methods

    @staticmethod
    def unwrap_component(component: Any) -> Any:
        """Unwrap ClassComponent/FunctionComponent wrapper if present.

        Args
        ----
        component : Any
            Potentially wrapped component.

        Returns
        -------
        Any
            Unwrapped component.
        """
        if hasattr(component, COMPONENT_VALUE_ATTR):
            return getattr(component, COMPONENT_VALUE_ATTR)
        return component

    @staticmethod
    def get_implements_port(component: Any) -> str | None:
        """Extract the port implementation declaration from a component.

        Args
        ----
        component : Any
            Component to extract from (may be wrapped).

        Returns
        -------
        str | None
            Port name if declared, None otherwise.
        """
        unwrapped = RegistryValidator.unwrap_component(component)
        if hasattr(unwrapped, IMPLEMENTS_PORT_ATTR):
            port: str = getattr(unwrapped, IMPLEMENTS_PORT_ATTR)
            return port
        return None

    @staticmethod
    def get_required_ports(component: Any) -> list[str]:
        """Extract the list of required ports from a component.

        Args
        ----
        component : Any
            Component to extract from (may be wrapped).

        Returns
        -------
        list[str]
            List of required port names.
        """
        unwrapped = RegistryValidator.unwrap_component(component)
        if hasattr(unwrapped, REQUIRED_PORTS_ATTR):
            return getattr(unwrapped, REQUIRED_PORTS_ATTR, [])
        return []

    # Validation methods

    @staticmethod
    def validate_component_type(component_type: str) -> ComponentType:
        """Validate and convert component type string to enum.

        Args
        ----
        component_type : str
            Component type as string.

        Returns
        -------
        ComponentType
            Valid ComponentType enum value.

        Raises
        ------
        InvalidComponentError
            If component type is invalid.
        """
        try:
            return ComponentType(component_type)
        except ValueError as e:
            valid = ", ".join(t.value for t in ComponentType)
            raise InvalidComponentError(
                component_type, f"Invalid component type. Must be one of: {valid}"
            ) from e

    @staticmethod
    def validate_component_name(name: str) -> None:
        """Validate component name format.

        Args
        ----
        name : str
            Component name to validate.

        Raises
        ------
        InvalidComponentError
            If name is invalid.
        """
        if not name:
            raise InvalidComponentError("<empty>", "Component name must be a non-empty string")

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise InvalidComponentError(name, f"Component name must be alphanumeric, got '{name}'")

    @staticmethod
    def validate_namespace(namespace: str | None) -> str:
        """Validate and normalize namespace (defaults to 'user').

        Args
        ----
        namespace : str | None
            Namespace to validate (None defaults to "user").

        Returns
        -------
        str
            Normalized namespace string.

        Raises
        ------
        InvalidComponentError
            If namespace contains invalid characters.
        """
        if namespace is None or namespace == "":
            return Namespace.USER  # Default namespace

        if not re.match(r"^[a-zA-Z0-9_]+$", namespace):
            raise InvalidComponentError(
                namespace, f"Namespace must be alphanumeric, got '{namespace}'"
            )

        return namespace.lower()  # Normalize to lowercase

    @staticmethod
    def wrap_component(
        component: object,
    ) -> ClassComponent | FunctionComponent | InstanceComponent:
        """Wrap raw component in appropriate type wrapper.

        Args
        ----
        component : object
            Raw component to wrap.

        Returns
        -------
        ClassComponent | FunctionComponent | InstanceComponent
            Wrapped component.
        """
        if inspect.isclass(component):
            return ClassComponent(value=component)
        if inspect.isfunction(component) or inspect.ismethod(component):
            return FunctionComponent(value=component)
        return InstanceComponent(value=component)

    @classmethod
    def is_protected_namespace(cls, namespace: str) -> bool:
        """Check if namespace is protected and requires privileges.

        Args
        ----
        namespace : str
            Namespace to check.

        Returns
        -------
        bool
            True if namespace is protected.
        """
        return namespace in cls.PROTECTED_NAMESPACES
