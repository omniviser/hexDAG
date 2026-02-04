"""Documentation extractors for hexDAG components.

This module provides utilities to extract documentation from code artifacts
including signatures, docstrings, decorators, and explicit schema attributes.
"""

import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from hexdag.core.docs.models import (
    AdapterDoc,
    NodeDoc,
    ParameterDoc,
    ToolDoc,
)
from hexdag.core.logging import get_logger
from hexdag.core.schema import SchemaGenerator
from hexdag.core.secrets import SecretDescriptor

logger = get_logger(__name__)


class DocExtractor:
    """Extract documentation from Python code artifacts.

    This class provides static methods to extract structured documentation
    from adapters, nodes, tools, and other callables by inspecting their
    signatures, docstrings, and special attributes.
    """

    @staticmethod
    def extract_parameters(obj: Callable | type) -> list[ParameterDoc]:
        """Extract parameter documentation from a callable or class __init__.

        Parameters
        ----------
        obj : Callable | type
            Function, method, or class to extract parameters from

        Returns
        -------
        list[ParameterDoc]
            List of documented parameters
        """
        # Get the target to inspect
        if isinstance(obj, type):
            target = obj.__init__
        elif callable(obj) and not inspect.isfunction(obj):
            target = obj.__call__
        else:
            target = obj

        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not get signature for {obj}: {e}")
            return []

        # Extract docstring parameter descriptions
        param_docs = SchemaGenerator._extract_param_docs(target)

        # Try to get type hints
        try:
            hints = get_type_hints(target)
        except Exception:
            hints = {}

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "args", "kwargs"):
                continue

            # Skip VAR_POSITIONAL and VAR_KEYWORD
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # Get type hint
            type_hint = hints.get(param_name, param.annotation)
            if type_hint == inspect.Parameter.empty:
                type_hint_str = "Any"
            else:
                type_hint_str = DocExtractor._format_type_hint(type_hint)

            # Check for secret descriptor
            default_value = param.default
            is_secret = isinstance(default_value, SecretDescriptor)

            # Get default value
            if default_value == inspect.Parameter.empty:
                default_str = None
                required = True
            elif is_secret:
                default_str = f"secret(env='{default_value.env_var}')"
                required = default_value.required
            else:
                default_str = repr(default_value)
                required = False

            # Get description from docstring
            description = param_docs.get(param_name, "")

            # Check for enum/Literal values
            enum_values = DocExtractor._extract_enum_values(type_hint)

            parameters.append(
                ParameterDoc(
                    name=param_name,
                    type_hint=type_hint_str,
                    description=description,
                    required=required,
                    default=default_str,
                    enum_values=enum_values,
                )
            )

        return parameters

    @staticmethod
    def _format_type_hint(hint: Any) -> str:
        """Format a type hint as a readable string.

        Parameters
        ----------
        hint : Any
            Type annotation

        Returns
        -------
        str
            Human-readable type string
        """
        if hint is None or hint is type(None):
            return "None"

        # Handle string annotations
        if isinstance(hint, str):
            return hint

        # Get origin and args for generic types
        origin = getattr(hint, "__origin__", None)
        args = getattr(hint, "__args__", ())

        # Handle Union types (including | syntax)
        if origin is type(int | str):  # UnionType
            parts = [DocExtractor._format_type_hint(arg) for arg in args]
            return " | ".join(parts)

        # Handle Optional (Union with None)
        try:
            from typing import Union

            if origin is Union:
                non_none = [arg for arg in args if arg is not type(None)]
                if len(non_none) == 1 and type(None) in args:
                    return f"{DocExtractor._format_type_hint(non_none[0])} | None"
                parts = [DocExtractor._format_type_hint(arg) for arg in args]
                return " | ".join(parts)
        except ImportError:
            pass

        # Handle Literal
        try:
            from typing import Literal, get_origin

            if get_origin(hint) is Literal:
                values = ", ".join(repr(v) for v in args)
                return f"Literal[{values}]"
        except (ImportError, TypeError):
            pass

        # Handle list, dict, etc.
        if origin is list:
            if args:
                return f"list[{DocExtractor._format_type_hint(args[0])}]"
            return "list"
        if origin is dict:
            if len(args) == 2:
                key_type = DocExtractor._format_type_hint(args[0])
                val_type = DocExtractor._format_type_hint(args[1])
                return f"dict[{key_type}, {val_type}]"
            return "dict"
        if origin is set:
            if args:
                return f"set[{DocExtractor._format_type_hint(args[0])}]"
            return "set"
        if origin is tuple:
            if args:
                parts = [DocExtractor._format_type_hint(arg) for arg in args]
                return f"tuple[{', '.join(parts)}]"
            return "tuple"

        # Handle basic types
        if hasattr(hint, "__name__"):
            return hint.__name__

        # Fallback
        return str(hint)

    @staticmethod
    def _extract_enum_values(hint: Any) -> list[str] | None:
        """Extract enum values from Literal type hints.

        Parameters
        ----------
        hint : Any
            Type annotation to check

        Returns
        -------
        list[str] | None
            List of allowed values or None
        """
        try:
            from typing import Literal, get_args, get_origin

            if get_origin(hint) is Literal:
                return [str(v) for v in get_args(hint)]
        except (ImportError, TypeError):
            pass
        return None

    @staticmethod
    def extract_docstring_parts(obj: Any) -> tuple[str, str, list[str]]:
        """Extract description, full docstring, and examples from docstring.

        Parameters
        ----------
        obj : Any
            Object with __doc__ attribute

        Returns
        -------
        tuple[str, str, list[str]]
            (first_line_description, full_docstring, examples)
        """
        docstring = inspect.getdoc(obj) or ""
        if not docstring:
            return "", "", []

        # First line as description
        lines = docstring.split("\n")
        description = lines[0].strip() if lines else ""

        # Extract examples
        examples = DocExtractor._extract_examples(docstring)

        return description, docstring, examples

    @staticmethod
    def _extract_examples(docstring: str) -> list[str]:
        """Extract code examples from docstring.

        Parameters
        ----------
        docstring : str
            Full docstring text

        Returns
        -------
        list[str]
            List of code examples
        """
        examples = []

        # Find Examples section
        patterns = [
            r"Examples?\s*[-=]*\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s*[-=]*\s*\n|\Z)",
            r"Examples?\s*\n\s*-+\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s*\n\s*-+|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)
            if match:
                examples_text = match.group(1)
                # Extract code blocks
                code_blocks = re.findall(
                    r"```(?:python)?\n(.*?)```|>>> (.*?)(?=\n(?!\.\.\.|\s)|\Z)",
                    examples_text,
                    re.DOTALL,
                )
                for block in code_blocks:
                    code = block[0] or block[1]
                    if code.strip():
                        examples.append(code.strip())
                break

        return examples

    @staticmethod
    def extract_adapter_doc(cls: type) -> AdapterDoc:
        """Extract documentation from an adapter class.

        Parameters
        ----------
        cls : type
            Adapter class to document

        Returns
        -------
        AdapterDoc
            Extracted adapter documentation
        """
        description, full_docstring, examples = DocExtractor.extract_docstring_parts(cls)
        parameters = DocExtractor.extract_parameters(cls)

        # Determine port type from class name or protocol
        port_type = DocExtractor._guess_port_type(cls)

        # Extract secrets from signature
        secrets = DocExtractor._extract_secrets(cls)

        # Generate module path
        module_path = f"{cls.__module__}.{cls.__name__}"

        # Generate decorator example
        decorator_example = DocExtractor._generate_adapter_decorator_example(
            cls, port_type, secrets
        )

        # Generate YAML example
        yaml_example = DocExtractor._generate_adapter_yaml_example(cls, port_type)

        return AdapterDoc(
            name=cls.__name__,
            module_path=module_path,
            description=description,
            full_docstring=full_docstring,
            parameters=parameters,
            examples=examples,
            yaml_example=yaml_example,
            port_type=port_type,
            secrets=secrets,
            decorator_example=decorator_example,
        )

    @staticmethod
    def _guess_port_type(cls: type) -> str:
        """Guess port type from adapter class name or implemented protocols.

        Parameters
        ----------
        cls : type
            Adapter class

        Returns
        -------
        str
            Guessed port type
        """
        name_lower = cls.__name__.lower()

        # Check class name patterns
        if "llm" in name_lower or "openai" in name_lower or "anthropic" in name_lower:
            return "llm"
        if "memory" in name_lower:
            return "memory"
        if "database" in name_lower or "sql" in name_lower:
            return "database"
        if "secret" in name_lower or "keyvault" in name_lower:
            return "secret"
        if "storage" in name_lower or "blob" in name_lower:
            return "storage"
        if "tool" in name_lower and "router" in name_lower:
            return "tool_router"
        if "embedding" in name_lower:
            return "embedding"
        if "observer" in name_lower:
            return "observer_manager"
        if "policy" in name_lower:
            return "policy_manager"

        # Check implemented protocols via base classes
        for base in cls.__mro__:
            base_name = base.__name__.lower()
            if "generation" in base_name or "llm" in base_name:
                return "llm"
            if "memory" in base_name:
                return "memory"
            if "database" in base_name:
                return "database"

        return "unknown"

    @staticmethod
    def _extract_secrets(cls: type) -> dict[str, str]:
        """Extract secret declarations from __init__ signature.

        Parameters
        ----------
        cls : type
            Class to inspect

        Returns
        -------
        dict[str, str]
            Mapping of parameter name to environment variable name
        """
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            return {}

        secrets = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            if isinstance(param.default, SecretDescriptor):
                secrets[param_name] = param.default.env_var

        return secrets

    @staticmethod
    def _generate_adapter_decorator_example(
        cls: type, port_type: str, secrets: dict[str, str]
    ) -> str:
        """Generate example @adapter decorator usage.

        Parameters
        ----------
        cls : type
            Adapter class
        port_type : str
            Port type
        secrets : dict[str, str]
            Secret mappings

        Returns
        -------
        str
            Example decorator code
        """
        name = cls.__name__.replace("Adapter", "").lower()

        if secrets:
            secrets_str = ", ".join(f'"{k}": "{v}"' for k, v in secrets.items())
            return f'@adapter("{port_type}", name="{name}", secrets={{{secrets_str}}})'
        return f'@adapter("{port_type}", name="{name}")'

    @staticmethod
    def _generate_adapter_yaml_example(cls: type, port_type: str) -> str:
        """Generate YAML usage example for adapter.

        Parameters
        ----------
        cls : type
            Adapter class
        port_type : str
            Port type

        Returns
        -------
        str
            YAML example
        """
        module_path = f"{cls.__module__}.{cls.__name__}"
        return f"""ports:
  {port_type}:
    adapter: {module_path}
    config:
      # Add configuration here"""

    @staticmethod
    def extract_node_doc(cls: type) -> NodeDoc:
        """Extract documentation from a node factory class.

        Parameters
        ----------
        cls : type
            Node factory class to document

        Returns
        -------
        NodeDoc
            Extracted node documentation
        """
        description, full_docstring, examples = DocExtractor.extract_docstring_parts(cls)

        # Check for _yaml_schema
        yaml_schema = getattr(cls, "_yaml_schema", None)

        # If has explicit schema, extract parameters from it
        if yaml_schema and isinstance(yaml_schema, dict):
            parameters = DocExtractor._extract_params_from_schema(yaml_schema)
            if "description" in yaml_schema:
                description = yaml_schema["description"]
        else:
            # Extract from __call__ method
            try:
                instance = cls()
                parameters = DocExtractor.extract_parameters(instance)
            except Exception:
                parameters = DocExtractor.extract_parameters(cls)

        # Generate module path
        module_path = f"{cls.__module__}.{cls.__name__}"

        # Determine kind from class name
        kind = DocExtractor._class_name_to_kind(cls.__name__)

        # Generate YAML example
        yaml_example = DocExtractor._generate_node_yaml_example(cls, kind, yaml_schema)

        return NodeDoc(
            name=cls.__name__,
            module_path=module_path,
            description=description,
            full_docstring=full_docstring,
            parameters=parameters,
            examples=examples,
            yaml_example=yaml_example,
            namespace="core",
            yaml_schema=yaml_schema,
            kind=kind,
        )

    @staticmethod
    def _class_name_to_kind(class_name: str) -> str:
        """Convert class name to YAML kind.

        Parameters
        ----------
        class_name : str
            Class name (e.g., "LLMNode", "FunctionNode")

        Returns
        -------
        str
            YAML kind (e.g., "llm_node", "function_node")
        """
        # Split on uppercase letters and join with underscore
        words = re.findall(r"[A-Z][a-z]*", class_name)
        return "_".join(word.lower() for word in words)

    @staticmethod
    def _extract_params_from_schema(schema: dict[str, Any]) -> list[ParameterDoc]:
        """Extract parameters from _yaml_schema.

        Parameters
        ----------
        schema : dict[str, Any]
            JSON Schema dict

        Returns
        -------
        list[ParameterDoc]
            Extracted parameters
        """
        parameters = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            param_type = prop_schema.get("type", "any")
            description = prop_schema.get("description", "")
            default = prop_schema.get("default")

            # Handle enum
            enum_values = prop_schema.get("enum")
            if enum_values:
                enum_values = [str(v) for v in enum_values]

            parameters.append(
                ParameterDoc(
                    name=prop_name,
                    type_hint=param_type,
                    description=description,
                    required=prop_name in required,
                    default=repr(default) if default is not None else None,
                    enum_values=enum_values,
                )
            )

        return parameters

    @staticmethod
    def _generate_node_yaml_example(
        cls: type, kind: str, yaml_schema: dict[str, Any] | None
    ) -> str:
        """Generate YAML usage example for node.

        Parameters
        ----------
        cls : type
            Node class
        kind : str
            Node kind
        yaml_schema : dict[str, Any] | None
            Explicit schema if available

        Returns
        -------
        str
            YAML example
        """
        if yaml_schema:
            return SchemaGenerator.generate_example_yaml(kind, yaml_schema)

        # Fallback basic example
        return f"""- kind: {kind}
  metadata:
    name: my_{kind.replace("_node", "")}
  spec:
    # Add configuration here
  dependencies: []"""

    @staticmethod
    def extract_tool_doc(func: Callable) -> ToolDoc:
        """Extract documentation from a tool function.

        Parameters
        ----------
        func : Callable
            Tool function to document

        Returns
        -------
        ToolDoc
            Extracted tool documentation
        """
        description, full_docstring, examples = DocExtractor.extract_docstring_parts(func)
        parameters = DocExtractor.extract_parameters(func)

        # Get return type
        try:
            hints = get_type_hints(func)
            return_type = hints.get("return")
            return_type_str = DocExtractor._format_type_hint(return_type) if return_type else "Any"
        except Exception:
            return_type_str = "Any"

        # Check if async
        is_async = inspect.iscoroutinefunction(func)

        # Generate module path
        module_path = f"{func.__module__}.{func.__name__}"

        # Generate YAML example for agent usage
        yaml_example = DocExtractor._generate_tool_yaml_example(func)

        return ToolDoc(
            name=func.__name__,
            module_path=module_path,
            description=description,
            full_docstring=full_docstring,
            parameters=parameters,
            examples=examples,
            yaml_example=yaml_example,
            namespace="core",
            return_type=return_type_str,
            is_async=is_async,
        )

    @staticmethod
    def _generate_tool_yaml_example(func: Callable) -> str:
        """Generate YAML example showing tool usage with agents.

        Parameters
        ----------
        func : Callable
            Tool function

        Returns
        -------
        str
            YAML example
        """
        module_path = f"{func.__module__}.{func.__name__}"
        return f"""- kind: agent_node
  metadata:
    name: my_agent
  spec:
    tools:
      - {module_path}
    # ... other config"""
