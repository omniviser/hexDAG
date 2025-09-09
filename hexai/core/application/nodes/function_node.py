"""Function node factory for creating function-based pipeline nodes."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from ...domain.dag import NodeSpec
from .base_node_factory import BaseNodeFactory


class FunctionNode(BaseNodeFactory):
    """Simple factory for creating function-based nodes with optional Pydantic validation."""

    def __call__(
        self,
        name: str,
        fn: Callable[..., Any],
        input_schema: dict[str, Any] | type[BaseModel] | None = None,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
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
            input_mapping: Optional field mapping dict {target_field: source_path}
            **kwargs: Additional parameters
        """
        # Validate function can be used properly
        self._validate_function(fn)
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
            input_model: type[BaseModel] | type | None = input_schema
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
            output_model: type[BaseModel] | type | None = output_schema
        else:
            output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Create the wrapped function
        wrapped_fn = self._create_wrapped_function(name, fn, input_model, output_model)

        # Add input_mapping to params if provided (including empty dict)
        params = kwargs.copy()
        if input_mapping is not None:
            params["input_mapping"] = input_mapping

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_type=input_model,
            out_type=output_model,
            deps=set(deps or []),
            params=params,
        )

    def _create_wrapped_function(
        self,
        name: str,
        fn: Callable[..., Any],
        input_model: type[BaseModel] | type | None,
        output_model: type[BaseModel] | type | None,
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
    ) -> tuple[type[BaseModel] | None, type[BaseModel] | None]:
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
        """Create a passthrough mapping where field names are preserved.

        Args
        ----
            fields: List of field names to create passthrough mapping for

        Returns
        -------
            Dict mapping each field to itself

        Example
        -------
            create_passthrough_mapping(["text", "score"])
            # Returns: {"text": "text", "score": "score"}
        """
        return {field: field for field in fields}

    @staticmethod
    def create_rename_mapping(field_mapping: dict[str, str]) -> dict[str, str]:
        """Create a rename mapping from source to target field names.

        Args
        ----
            field_mapping: Dict mapping target field names to source field names

        Returns
        -------
            The same mapping (for consistency with other mapping methods)

        Example
        -------
            create_rename_mapping({"content": "text", "validation": "status"})
            # Returns: {"content": "text", "validation": "status"}
        """
        return field_mapping.copy()

    @staticmethod
    def create_prefixed_mapping(
        fields: list[str], source_node: str, prefix: str = ""
    ) -> dict[str, str]:
        """Create a mapping that prefixes field names and maps to a source node.

        Args
        ----
            fields: List of field names to map
            source_node: Name of the source node to map from
            prefix: Prefix to add to field names (default: "")

        Returns
        -------
            Dict mapping prefixed field names to source node paths

        Example
        -------
            create_prefixed_mapping(["text", "score"], "processor", "proc_")
            # Returns: {"proc_text": "processor.text", "proc_score": "processor.score"}
        """
        return {f"{prefix}{field}": f"{source_node}.{field}" for field in fields}

    def with_input_mapping(self, node_spec: NodeSpec, input_mapping: dict[str, str]) -> NodeSpec:
        """Enhance an existing NodeSpec with input mapping.

        Args
        ----
            node_spec: Existing NodeSpec to enhance
            input_mapping: Input mapping to add

        Returns
        -------
            New NodeSpec with input mapping added to params
        """
        # Create a copy of the existing params
        new_params = node_spec.params.copy()
        new_params["input_mapping"] = input_mapping

        # Return new NodeSpec with updated params
        return NodeSpec(
            name=node_spec.name,
            fn=node_spec.fn,
            in_type=node_spec.in_type,
            out_type=node_spec.out_type,
            deps=node_spec.deps,
            params=new_params,
        )
