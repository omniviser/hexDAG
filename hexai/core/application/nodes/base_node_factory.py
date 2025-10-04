"""Simplified BaseNodeFactory for creating nodes with Pydantic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, create_model

from hexai.core.domain.dag import NodeSpec
from hexai.core.protocols import is_schema_type

if TYPE_CHECKING:
    from hexai.core.application.prompt.template import PromptTemplate


@lru_cache(maxsize=256)
def _create_cached_model(name: str, fields_tuple: tuple[tuple[str, Any], ...]) -> type[BaseModel]:
    """Create and cache Pydantic models using lru_cache.

    Uses hash-based lookup (faster than string keys) with automatic LRU eviction.
    Thread-safe and prevents unbounded cache growth.

    Cache key includes both name and fields_tuple to prevent collisions when
    different schemas share the same model name.

    Parameters
    ----------
    name : str
        Model class name
    fields_tuple : tuple[tuple[str, Any], ...]
        Hashable field definitions (used in cache key to ensure uniqueness)

    Returns
    -------
    type[BaseModel]
        Cached Pydantic model class

    Notes
    -----
    The cache key is the combination of (name, fields_tuple), so two different
    field definitions with the same name will create separate cache entries.
    This prevents cache collisions while still providing performance benefits.
    """
    fields_dict = dict(fields_tuple)
    return create_model(name, **fields_dict)


class BaseNodeFactory(ABC):
    """Minimal base class for node factories with Pydantic models."""

    # Note: Event emission has been removed as it's now handled by the orchestrator
    # The new event system uses ObserverManager at the orchestrator level

    def create_pydantic_model(
        self, name: str, schema: dict[str, Any] | type[BaseModel] | type[Any] | None
    ) -> type[BaseModel] | None:
        """Create a Pydantic model from a schema.

        Raises
        ------
        ValueError
            If schema type is not supported
        """
        if schema is None:
            return None

        if is_schema_type(schema):
            return schema  # type: ignore[return-value]  # is_schema_type checks for BaseModel subclass

        if isinstance(schema, dict):
            # String type names mapping (for when field_type is a string)
            type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "Any": Any,
            }

            # Create field definitions for create_model
            field_definitions: dict[str, Any] = {}
            for field_name, field_type in schema.items():
                # Dispatch based on field_type's type using match pattern
                match field_type:
                    case str():
                        # String type names - convert to actual types
                        actual_type = type_map.get(field_type, Any)
                        field_definitions[field_name] = (actual_type, ...)
                    case type():
                        # Already a type
                        field_definitions[field_name] = (field_type, ...)
                    case tuple():
                        # Already in the correct format (type, default)
                        field_definitions[field_name] = field_type
                    case _:
                        # Unknown type specification - use Any
                        field_definitions[field_name] = (Any, ...)

            # Convert to tuple for lru_cache (hashable)
            fields_tuple = tuple(sorted(field_definitions.items()))
            return _create_cached_model(name, fields_tuple)

        # Handle primitive types - create a simple wrapper model
        # At this point, schema should be a type
        try:
            return cast("type[Any] | None", create_model(name, value=(schema, ...)))
        except (TypeError, AttributeError) as e:
            # If we get here, schema is an unexpected type
            raise ValueError(
                f"Schema must be a dict, type, or Pydantic model, got {type(schema).__name__}"
            ) from e

    @staticmethod
    def infer_input_schema_from_template(
        template: str | PromptTemplate,
        special_params: set[str] | None = None,
    ) -> dict[str, Any]:
        """Infer input schema from template variables with configurable filtering.

        This method extracts variable names from a prompt template and creates
        a schema dictionary mapping those variables to string types. It supports
        filtering out special parameters that are not user inputs.

        Parameters
        ----------
        template : str | PromptTemplate
            The prompt template to analyze. Can be a string or PromptTemplate instance.
        special_params : set[str] | None, optional
            Set of parameter names to exclude from the schema (e.g., "context_history").
            If None, no filtering is applied.

        Returns
        -------
        dict[str, Any]
            Schema dictionary mapping variable names to str type.
            Returns {"input": str} if no variables found.

        Examples
        --------
        >>> BaseNodeFactory.infer_input_schema_from_template("Hello {{name}}")
        {'name': <class 'str'>}

        >>> BaseNodeFactory.infer_input_schema_from_template(
        ...     "Process {{user}} with {{context_history}}",
        ...     special_params={"context_history"}
        ... )
        {'user': <class 'str'>}

        >>> BaseNodeFactory.infer_input_schema_from_template("No variables")
        {'input': <class 'str'>}
        """
        # Import here to avoid circular dependency
        from hexai.core.application.prompt.template import PromptTemplate

        # Convert string to PromptTemplate if needed
        if isinstance(template, str):
            template = PromptTemplate(template)

        # Extract variables from template
        variables = getattr(template, "input_vars", [])

        # Filter special parameters if provided
        if special_params:
            variables = [v for v in variables if v not in special_params]

        # Return default schema if no variables
        if not variables:
            return {"input": str}

        # Handle nested variables (e.g., "user.name" -> "user")
        schema: dict[str, Any] = {}
        for var in variables:
            base_var = var.split(".")[0]
            # Double-check against special params for nested variables
            if not special_params or base_var not in special_params:
                schema[base_var] = str

        return schema

    def _copy_required_ports_to_wrapper(self, wrapper_fn: Any) -> None:
        """Copy required_ports metadata from factory class to wrapper function.

        This ensures port requirements are preserved when creating node functions.
        """
        if hasattr(self.__class__, "_hexdag_required_ports"):
            # _hexdag_required_ports is added dynamically by @node decorator
            wrapper_fn._hexdag_required_ports = self.__class__._hexdag_required_ports  # pyright: ignore[reportAttributeAccessIssue]  # noqa: B010

    def create_node_with_mapping(
        self,
        name: str,
        wrapped_fn: Any,
        input_schema: dict[str, Any] | None,
        output_schema: dict[str, Any] | type[BaseModel] | None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Universal NodeSpec creation."""
        # Copy required_ports metadata to wrapper
        self._copy_required_ports_to_wrapper(wrapped_fn)

        # Create Pydantic models
        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=kwargs,
        )

    @abstractmethod
    def __call__(self, name: str, *args: Any, **kwargs: Any) -> NodeSpec:  # noqa: ARG002
        """Create a NodeSpec.

        Must be implemented by subclasses.

        Args:
            name: Name of the node
            *args: Additional positional arguments (unused, for subclass flexibility)
            **kwargs: Additional keyword arguments
        """
        _ = args  # Marked as intentionally unused for subclass API flexibility
        raise NotImplementedError
