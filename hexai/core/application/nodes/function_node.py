"""Function node factory for creating function-based pipeline nodes."""

import asyncio
import inspect
from typing import Any, Callable, Type, get_type_hints

from pydantic import BaseModel

from ...domain.dag import NodeSpec
from .base_node_factory import BaseNodeFactory
from .mapped_input import MappedInput


class FunctionNode(BaseNodeFactory):
    """Simple factory for creating function-based nodes with optional Pydantic validation."""

    def __call__(
        self,
        name: str,
        fn: Callable[..., Any],
        input_schema: dict[str, Any] | Type[BaseModel] | None = None,
        output_schema: dict[str, Any] | Type[BaseModel] | None = None,
        deps: list[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a function-based node.

        Args:
        ----
            name: Node name
            fn: Function to execute
            input_schema: Input schema for validation (if None, inferred from function)
            output_schema: Output schema for validation (if None, inferred from function)
            deps: List of dependency node names
            input_mapping: Optional field mapping dict {target_field: "source.path"}
            **kwargs: Additional parameters
        """
        # Validate function can be used properly
        self._validate_function(fn)

        # Store input_mapping in kwargs if provided
        if input_mapping is not None:
            kwargs["input_mapping"] = input_mapping

        # Handle input_mapping: auto-generate input schema if mapping provided
        if input_mapping and not input_schema:
            # Auto-generate Pydantic model from field mapping
            input_schema = MappedInput.create_model(
                f"{name}MappedInput",
                input_mapping,
                dependency_models=None,  # Could enhance with dependency introspection
            )

        # Infer schemas from function annotations if not provided
        if input_schema is None or output_schema is None:
            inferred_input, inferred_output = self._infer_schemas_from_function(fn)
            input_schema = input_schema or inferred_input
            output_schema = output_schema or inferred_output

        # Create Pydantic models for validation
        # For basic types like dict, list, str, etc., use them directly

        # Handle input schema - check if it's a basic Python type
        if isinstance(input_schema, type) and input_schema.__name__ in {
            "dict",
            "list",
            "str",
            "int",
            "float",
            "bool",
        }:
            input_model: Type[BaseModel] | type | None = input_schema
        else:
            input_model = self.create_pydantic_model(f"{name}Input", input_schema)

        # Handle output schema - check if it's a basic Python type
        if isinstance(output_schema, type) and output_schema.__name__ in {
            "dict",
            "list",
            "str",
            "int",
            "float",
            "bool",
        }:
            output_model: Type[BaseModel] | type | None = output_schema
        else:
            output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Create the wrapped function
        wrapped_fn = self._create_wrapped_function(name, fn, input_model, output_model)

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=set(deps or []),
            params=kwargs,
        )

    def _create_wrapped_function(
        self,
        name: str,
        fn: Callable[..., Any],
        input_model: Type[BaseModel] | type | None,
        output_model: Type[BaseModel] | type | None,
    ) -> Callable[..., Any]:
        """Create a simple wrapped function with explicit port handling."""
        # Analyze function signature once
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        param_names = set(sig.parameters.keys())

        async def wrapped_fn(input_data: Any, **ports: Any) -> Any:
            """Execute function with explicit port handling."""
            # Orchestrator handles all validation now - just execute the function

            # Prepare function call arguments
            if accepts_kwargs:
                # Function accepts **kwargs, pass all ports
                call_kwargs = ports
            else:
                # Function has specific parameters, only pass ports that match parameter names
                call_kwargs = {k: v for k, v in ports.items() if k in param_names}

            # Execute function (handle both sync and async)
            if asyncio.iscoroutinefunction(fn):
                result = await fn(input_data, **call_kwargs)
            else:
                result = fn(input_data, **call_kwargs)

            # Return raw result - orchestrator will handle output validation
            return result

        # Preserve function metadata
        wrapped_fn.__name__ = getattr(fn, "__name__", f"wrapped_{name}")
        wrapped_fn.__doc__ = getattr(fn, "__doc__", f"Wrapped function: {name}")

        return wrapped_fn

    def _validate_function(self, fn: Callable[..., Any]) -> None:
        """Validate that function can be properly wrapped.

        Args
        ----
            fn: Function to validate

        Raises
        ------
            ValueError: If function cannot be used
        """
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # Function needs at least one parameter to receive input_data
        if not params:
            raise ValueError("Function must have at least one parameter to receive input_data")

        # Check if first parameter is suitable (not **kwargs only)
        first_param = params[0]
        if first_param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("First parameter cannot be **kwargs - need parameter for input_data")

    def _infer_schemas_from_function(
        self, fn: Callable[..., Any]
    ) -> tuple[Type[BaseModel] | None, Type[BaseModel] | None]:
        """Infer input and output schemas from function type annotations.

        Args
        ----
            fn: Function to analyze

        Returns
        -------
            Tuple of (input_schema, output_schema) where each can be None if not inferrable
        """
        try:
            # Get type hints
            type_hints = get_type_hints(fn)
            sig = inspect.signature(fn)

            # Infer input schema from first parameter (skip 'self' if present)
            input_schema = None
            params = list(sig.parameters.values())
            if params:
                # Skip 'self' parameter if present
                first_param = (
                    params[0]
                    if params[0].name != "self"
                    else (params[1] if len(params) > 1 else None)
                )
                if first_param and first_param.name in type_hints:
                    param_type = type_hints[first_param.name]
                    # Check if it's a Pydantic model
                    if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                        input_schema = param_type

            # Infer output schema from return annotation
            output_schema = None
            if "return" in type_hints:
                return_type = type_hints["return"]
                # Check if it's a Pydantic model
                if isinstance(return_type, type) and issubclass(return_type, BaseModel):
                    output_schema = return_type

            return input_schema, output_schema

        except Exception:
            # If anything goes wrong with inference, return None for both
            return None, None

    @staticmethod
    def create_passthrough_mapping(fields: list[str]) -> dict[str, str]:
        """Create a passthrough mapping where field names are unchanged.

        Args:
        ----
            fields: List of field names to pass through

        Returns
        -------
            Mapping dict {field: field} for each field
        """
        return {field: field for field in fields}

    @staticmethod
    def create_rename_mapping(mapping: dict[str, str]) -> dict[str, str]:
        """Create a simple rename mapping.

        Args:
        ----
            mapping: Dict of {new_name: old_name}

        Returns
        -------
            The mapping dict as-is (for consistency with other methods)
        """
        return mapping

    @staticmethod
    def create_prefixed_mapping(fields: list[str], source_node: str, prefix: str) -> dict[str, str]:
        """Create a mapping with prefixed field names.

        Args:
        ----
            fields: List of field names to map
            source_node: Name of the source node
            prefix: Prefix to add to field names

        Returns
        -------
            Mapping dict {prefix_field: source_node.field}
        """
        return {f"{prefix}{field}": f"{source_node}.{field}" for field in fields}

    def with_input_mapping(self, node: NodeSpec, input_mapping: dict[str, str]) -> NodeSpec:
        """Enhance an existing node with input mapping.

        Args:
        ----
            node: The node to enhance
            input_mapping: The input mapping to apply

        Returns
        -------
            New NodeSpec with the input mapping applied
        """
        # Create new params with the input_mapping
        new_params = dict(node.params) if node.params else {}
        new_params["input_mapping"] = input_mapping

        # Create new node with updated params
        return NodeSpec(
            name=node.name,
            fn=node.fn,
            in_model=node.in_model,
            out_model=node.out_model,
            deps=node.deps,
            params=new_params,
        )
