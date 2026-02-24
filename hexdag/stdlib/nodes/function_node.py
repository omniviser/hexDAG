"""Function node factory for creating function-based pipeline nodes."""

import asyncio
import importlib
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.logging import get_logger
from hexdag.kernel.protocols import is_schema_type
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory
from .mapped_input import MappedInput

logger = get_logger(__name__)


class FunctionNode(BaseNodeFactory):
    """Simple factory for creating function-based nodes with optional Pydantic validation.

    Function nodes are highly dynamic - the function itself defines configuration via its
    signature and parameters. No static Config class needed (follows YAGNI principle).
    All configuration is passed dynamically through __call__() parameters.
    """

    # Studio UI metadata
    _hexdag_icon = "Code"
    _hexdag_color = "#3b82f6"  # blue-500

    # Schema is auto-generated from __call__ signature by SchemaGenerator

    def __call__(
        self,
        name: str,
        fn: Callable[..., Any] | str,
        input_schema: dict[str, Any] | type[BaseModel] | None = None,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        unpack_input: bool = False,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a function-based node.

        Args:
        ----
            name: Node name
            fn: Function to execute (callable or module path string like 'mymodule.myfunc')
            input_schema: Input schema for validation (if None, inferred from function)
            output_schema: Output schema for validation (if None, inferred from function)
            deps: List of dependency node names
            input_mapping: Optional field mapping dict {target_field: "source.path"}
            unpack_input: If True, unpack input_mapping fields as individual **kwargs
                to the function instead of passing as single input_data object.
                This allows functions with signatures like `fn(load_id, rate, *, db=None)`
                instead of `fn(input_data, *, db=None)`.
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution
        """
        # Resolve function from string path if needed
        resolved_fn = self._resolve_function(fn)

        # Validate function can be used properly
        self._validate_function(resolved_fn, unpack_input=unpack_input)

        if input_mapping is not None:
            kwargs["input_mapping"] = input_mapping

        if unpack_input:
            kwargs["unpack_input"] = True

        if input_mapping and not input_schema:
            # Auto-generate Pydantic model from field mapping
            input_schema = MappedInput.create_model(
                f"{name}MappedInput",
                input_mapping,
                dependency_models=None,  # Could enhance with dependency introspection
            )

        # Infer schemas from function annotations if not provided
        if input_schema is None or output_schema is None:
            inferred_input, inferred_output = self._infer_schemas_from_function(resolved_fn)
            input_schema = input_schema or inferred_input
            output_schema = output_schema or inferred_output

        # For basic types like dict, list, str, etc., use them directly

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

        wrapped_fn = self._create_wrapped_function(
            name, resolved_fn, input_model, output_model, unpack_input=unpack_input
        )

        # Extract framework-level parameters from kwargs
        framework = self.extract_framework_params(kwargs)

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=kwargs,
            timeout=framework["timeout"],
            max_retries=framework["max_retries"],
            when=framework["when"],
        )

    def _create_wrapped_function(
        self,
        name: str,
        fn: Callable[..., Any],
        input_model: type[BaseModel] | type | None,
        output_model: type[BaseModel] | type | None,
        *,
        unpack_input: bool = False,
    ) -> Callable[..., Any]:
        """Create a simple wrapped function with explicit port handling.

        Parameters
        ----------
        name : str
            Node name for logging
        fn : Callable[..., Any]
            The function to wrap
        input_model : type[BaseModel] | type | None
            Input model for validation
        output_model : type[BaseModel] | type | None
            Output model for validation
        unpack_input : bool, default=False
            If True, unpack input_data fields as individual **kwargs to the function
            instead of passing as single input_data object.

        Returns
        -------
        Callable[..., Any]
            Wrapped function that handles orchestrator integration
        """
        # Analyze function signature once
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        param_names = set(sig.parameters.keys())

        # Get function name for logging
        fn_name = getattr(fn, "__name__", "anonymous")

        async def wrapped_fn(input_data: Any, **ports: Any) -> Any:
            """Execute function with explicit port handling."""
            node_logger = logger.bind(node=name, node_type="function_node")

            # Log input details at debug level
            if isinstance(input_data, dict):
                node_logger.debug(
                    "Input received",
                    input_keys=list(input_data.keys()),
                    input_key_count=len(input_data),
                )
            else:
                node_logger.debug(
                    "Input received",
                    input_type=type(input_data).__name__,
                )

            # Log execution start
            node_logger.info("Executing function", fn_name=fn_name)

            # Prepare function call arguments
            if accepts_kwargs:
                # Function accepts **kwargs, pass all ports
                call_kwargs = ports
            else:
                # Function has specific parameters, only pass ports that match parameter names
                call_kwargs = {k: v for k, v in ports.items() if k in param_names}

            with node_timer() as t:
                try:
                    # Execute function (handle both sync and async)
                    if unpack_input:
                        # Unpack input_data fields as individual kwargs
                        # This allows functions with signatures like fn(load_id, rate, *, db=None)
                        if isinstance(input_data, dict):
                            unpacked_kwargs = {**input_data, **call_kwargs}
                        elif isinstance(input_data, BaseModel):
                            unpacked_kwargs = {**input_data.model_dump(), **call_kwargs}
                        else:
                            # Fallback: try to convert to dict
                            unpacked_kwargs = {**vars(input_data), **call_kwargs}

                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(**unpacked_kwargs)
                        else:
                            result = fn(**unpacked_kwargs)
                    else:
                        # Standard behavior: pass input_data as first positional argument
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(input_data, **call_kwargs)
                        else:
                            result = fn(input_data, **call_kwargs)

                    node_logger.debug(
                        "Function completed",
                        fn_name=fn_name,
                        duration_ms=t.duration_str,
                        output_type=type(result).__name__,
                    )

                    return result

                except Exception as e:
                    node_logger.error(
                        "Function failed",
                        fn_name=fn_name,
                        duration_ms=t.duration_str,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

        # Preserve function metadata
        wrapped_fn.__name__ = getattr(fn, "__name__", f"wrapped_{name}")
        wrapped_fn.__doc__ = getattr(fn, "__doc__", f"Wrapped function: {name}")

        return wrapped_fn

    def _resolve_function(self, fn: Callable[..., Any] | str) -> Callable[..., Any]:
        """Resolve function from callable or module path string.

        Args
        ----
            fn: Function (callable) or module path string (e.g., 'mymodule.myfunc')

        Returns
        -------
        Callable[..., Any]
            The resolved callable function

        Raises
        ------
        TypeError
            If fn is not a callable or string
        ValueError
            If string path cannot be resolved to a callable
        """
        if callable(fn):
            return fn

        # At this point, fn should be a string based on type hints
        # But we validate at runtime for safety
        if not isinstance(fn, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"Expected a callable function or string module path, got {type(fn).__name__}"
            )

        # Parse module path string
        if "." not in fn:
            raise ValueError(f"Function path must be in format 'module.function', got: {fn}")

        # Split the module path
        module_path, func_name = fn.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            resolved_fn = getattr(module, func_name)

            if not callable(resolved_fn):
                raise ValueError(
                    f"Resolved '{fn}' is not callable (got {type(resolved_fn).__name__})"
                )

            return resolved_fn  # type: ignore[no-any-return]

        except ImportError as e:
            raise ValueError(f"Could not import module from function path '{fn}': {e}") from e
        except AttributeError as e:
            raise ValueError(f"Function '{func_name}' not found in module '{module_path}'") from e

    def _validate_function(self, fn: Callable[..., Any], *, unpack_input: bool = False) -> None:
        """Validate that function can be properly wrapped.

        Args
        ----
            fn: Function to validate
            unpack_input: If True, function receives unpacked kwargs instead of input_data

        Raises
        ------
        ValueError
            If function cannot be used
        """
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        if unpack_input:
            # With unpack_input, function receives fields as **kwargs
            # It can have any signature, including just **kwargs
            # No validation needed - fields are passed as keyword arguments
            return

        # Standard mode: function needs at least one parameter to receive input_data
        if not params:
            raise ValueError("Function must have at least one parameter to receive input_data")

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
        tuple[type[BaseModel] | None, type[BaseModel] | None]
            Tuple of (input_schema, output_schema) where each can be None if not inferrable
        """
        try:
            type_hints = get_type_hints(fn)
            sig = inspect.signature(fn)

            # Infer input schema from first parameter (skip 'self' if present)
            input_schema = None
            if params := list(sig.parameters.values()):
                # Skip 'self' parameter if present
                first_param = (
                    params[0]
                    if params[0].name != "self"
                    else (params[1] if len(params) > 1 else None)
                )
                if first_param and first_param.name in type_hints:
                    param_type = type_hints[first_param.name]
                    if is_schema_type(param_type):
                        input_schema = param_type

            # Infer output schema from return annotation
            output_schema = None
            if "return" in type_hints:
                return_type = type_hints["return"]
                if is_schema_type(return_type):
                    output_schema = return_type

            return input_schema, output_schema

        except (TypeError, AttributeError, ValueError):
            # If type hints are malformed or unavailable, skip inference
            return None, None

    @staticmethod
    def create_passthrough_mapping(fields: list[str]) -> dict[str, str]:
        """Create a passthrough mapping where field names are unchanged.

        Args:
        ----
            fields: List of field names to pass through

        Returns
        -------
        dict[str, str]
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
        dict[str, str]
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
        dict[str, str]
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
        NodeSpec
            New NodeSpec with the input mapping applied
        """
        new_params = dict(node.params) if node.params else {}
        new_params["input_mapping"] = input_mapping

        return NodeSpec(
            name=node.name,
            fn=node.fn,
            in_model=node.in_model,
            out_model=node.out_model,
            deps=node.deps,
            params=new_params,
            timeout=node.timeout,
            max_retries=node.max_retries,
            when=node.when,
        )
