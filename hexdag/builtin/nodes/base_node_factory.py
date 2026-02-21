"""Simplified BaseNodeFactory for creating nodes with Pydantic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from pydantic import BaseModel, create_model

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.prompt.template import PromptTemplate
from hexdag.core.protocols import is_schema_type
from hexdag.core.utils.caching import KeyedCache, schema_cache_key

# Module-level cache for dynamically created Pydantic models.
_MODEL_CACHE: KeyedCache[type[BaseModel]] = KeyedCache()

# String type names mapping used during model creation
_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "Any": Any,
}


class BaseNodeFactory(ABC):
    """Minimal base class for node factories with Pydantic models."""

    # Note: Event emission has been removed as it's now handled by the orchestrator
    # The new event system uses ObserverManager at the orchestrator level

    def create_pydantic_model(
        self, name: str, schema: dict[str, Any] | type[BaseModel] | type[Any] | None
    ) -> type[BaseModel] | None:
        """Create a Pydantic model from a schema.

        Uses a module-level cache to avoid recreating identical models.

        Raises
        ------
        ValueError
            If schema type is not supported
        """
        if schema is None:
            return None

        if is_schema_type(schema):
            # is_schema_type checks for BaseModel subclass
            return schema  # type: ignore[return-value]

        if isinstance(schema, dict):
            cache_key = (name, schema_cache_key(schema))

            def _build_model() -> type[BaseModel]:
                field_definitions: dict[str, Any] = {}
                for field_name, field_type in schema.items():
                    match field_type:
                        case str():
                            if field_type.endswith("?"):
                                base = field_type[:-1]
                                actual_type = _TYPE_MAP.get(base, Any)
                                field_definitions[field_name] = (actual_type | None, None)
                            else:
                                actual_type = _TYPE_MAP.get(field_type, Any)
                                field_definitions[field_name] = (actual_type, ...)
                        case type():
                            field_definitions[field_name] = (field_type, ...)
                        case tuple():
                            field_definitions[field_name] = field_type
                        case _:
                            field_definitions[field_name] = (Any, ...)
                return create_model(name, **field_definitions)

            return _MODEL_CACHE.get_or_create(cache_key, _build_model)

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

        if isinstance(template, str):
            template = PromptTemplate(template)

        variables = getattr(template, "input_vars", [])

        if special_params:
            variables = [v for v in variables if v not in special_params]

        if not variables:
            return {"input": str}

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
            wrapper_fn._hexdag_required_ports = (  # noqa: B010
                self.__class__._hexdag_required_ports  # pyright: ignore[reportAttributeAccessIssue]
            )

    @staticmethod
    def extract_framework_params(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract framework-level parameters from kwargs.

        This method provides a single place to extract NodeSpec framework params
        (timeout, max_retries, retry config, when) from kwargs. All node factories
        should use this to ensure consistent handling.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments to extract from (modified in place)

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: timeout, max_retries, retry_delay, retry_backoff,
            retry_max_delay, when. Values are None if not present in kwargs.

        Examples
        --------
        >>> kwargs = {
        ...     "when": "status == 'active'", "timeout": 30,
        ...     "max_retries": 3, "other": "value"
        ... }
        >>> framework = BaseNodeFactory.extract_framework_params(kwargs)
        >>> framework["timeout"]
        30
        >>> framework["max_retries"]
        3
        >>> kwargs
        {'other': 'value'}

        With retry backoff configuration::

        >>> kwargs = {"max_retries": 3, "retry_delay": 1.0, "retry_backoff": 2.0}
        >>> framework = BaseNodeFactory.extract_framework_params(kwargs)
        >>> framework["retry_delay"]
        1.0
        """
        return {
            "timeout": kwargs.pop("timeout", None),
            "max_retries": kwargs.pop("max_retries", None),
            "retry_delay": kwargs.pop("retry_delay", None),
            "retry_backoff": kwargs.pop("retry_backoff", None),
            "retry_max_delay": kwargs.pop("retry_max_delay", None),
            "when": kwargs.pop("when", None),
        }

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

        # Extract framework-level parameters from kwargs
        framework = self.extract_framework_params(kwargs)

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=kwargs,
            timeout=framework["timeout"],
            max_retries=framework["max_retries"],
            retry_delay=framework["retry_delay"],
            retry_backoff=framework["retry_backoff"],
            retry_max_delay=framework["retry_max_delay"],
            when=framework["when"],
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
