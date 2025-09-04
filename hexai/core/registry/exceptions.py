"""Custom exceptions for the registry system."""


class RegistryError(Exception):
    """Base exception for all registry-related errors."""

    pass


class ComponentNotFoundError(RegistryError):
    """Raised when a component cannot be found in the registry."""

    def __init__(self, name: str, namespace: str | None = None, available: list[str] | None = None):
        """Initialize the error with component details."""
        if namespace:
            msg = f"Component '{namespace}:{name}' not found"
        else:
            msg = f"Component '{name}' not found in any namespace"

        if available:
            msg += f". Available components: {', '.join(available[:5])}"
            if len(available) > 5:
                msg += f" ... and {len(available) - 5} more"

        super().__init__(msg)
        self.name = name
        self.namespace = namespace
        self.available = available


class NamespacePermissionError(RegistryError):
    """Raised when trying to register in a protected namespace without privilege."""

    def __init__(self, name: str, namespace: str = "core"):
        """Initialize with component name and namespace."""
        msg = (
            f"Cannot register '{name}' in protected '{namespace}' namespace. "
            f"Use a different namespace for plugin components."
        )
        super().__init__(msg)
        self.name = name
        self.namespace = namespace


class ComponentAlreadyRegisteredError(RegistryError):
    """Raised when trying to register a component that already exists."""

    def __init__(self, name: str, namespace: str):
        """Initialize with component details."""
        msg = f"Component '{namespace}:{name}' is already registered. Use replace=True to override."
        super().__init__(msg)
        self.name = name
        self.namespace = namespace


class InvalidComponentError(RegistryError):
    """Raised when trying to register an invalid component."""

    def __init__(self, name: str, reason: str):
        """Initialize with component name and reason."""
        msg = f"Cannot register '{name}': {reason}"
        super().__init__(msg)
        self.name = name
        self.reason = reason
