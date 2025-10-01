"""Configurable interface for plugins and adapters."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class ConfigurableComponent(Protocol):
    """Protocol for components that support configuration.

    Any adapter or plugin that implements this protocol will be
    automatically discoverable by the CLI config generation system.
    """

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class that defines configuration schema.

        Returns
        -------
        type[BaseModel]
            A Pydantic model class with field definitions, defaults, and validation
        """
        ...
