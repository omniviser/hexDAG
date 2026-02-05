"""!py YAML custom tag handler for inline Python functions.

This module provides a YAML custom tag constructor that compiles inline
Python code into callable functions, enabling inline Python in YAML pipelines.

Security Warning
----------------
The !py tag executes arbitrary Python code. Only use with trusted YAML files.
For untrusted input, use module path strings instead.

Examples
--------
YAML usage::

    body: !py |
      async def process(item, index, state, **ports):
          '''Process an item with database lookup.'''
          db = ports.get('database')
          if db:
              context = await db.aquery(item['id'])
          else:
              context = {}
          return {"item": item, "context": context}

The function will be compiled and made available as a callable.
"""

from collections.abc import Callable
from typing import Any

import yaml

from hexdag.core.logging import get_logger

logger = get_logger(__name__)


class PyTagError(Exception):
    """Error compiling !py tagged Python code."""

    pass


def py_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Callable[..., Any]:
    """Compile !py tagged Python code into a callable function.

    The !py block must define exactly one function. The function can be
    sync or async, and should follow the signature convention:

        async def process(item, index, state, **ports) -> Any:
            '''Docstring.'''
            ...

    Parameters
    ----------
    loader : yaml.SafeLoader
        YAML loader instance
    node : yaml.ScalarNode
        YAML node containing Python source code

    Returns
    -------
    Callable
        Compiled Python function

    Raises
    ------
    PyTagError
        If compilation fails or no function is defined

    Examples
    --------
    The following YAML::

        body: !py |
          def double(item, index, state, **ports):
              return item * 2

    Will compile to a callable function that can be invoked.
    """
    source_code = loader.construct_scalar(node)

    if not isinstance(source_code, str) or not source_code.strip():
        raise PyTagError("!py block must be a non-empty string defining a function.")

    # Compile the source code
    try:
        compiled = compile(source_code, "<yaml-!py>", "exec")
    except SyntaxError as e:
        raise PyTagError(f"Syntax error in !py block at line {e.lineno}: {e.msg}") from e
    except Exception as e:
        raise PyTagError(f"Failed to compile !py block: {e}") from e

    # Execute to get the function
    namespace: dict[str, Any] = {}
    try:
        exec(compiled, namespace)  # noqa: S102
    except Exception as e:
        raise PyTagError(f"Failed to execute !py block: {e}") from e

    # Find the defined function (first callable in namespace that's not a builtin)
    func: Callable[..., Any] | None = None
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if callable(obj) and not isinstance(obj, type):
            func = obj
            break

    if func is None:
        defined = [k for k in namespace if not k.startswith("_")]
        raise PyTagError(
            f"!py block must define a function. Found: {defined if defined else 'nothing'}"
        )

    logger.debug(
        "Compiled !py function",
        function_name=getattr(func, "__name__", "<anonymous>"),
    )

    return func


def validate_py_source(source_code: str) -> list[str]:
    """Validate !py source code without executing it.

    This function performs static validation of Python source code intended
    for !py tags. It checks syntax and verifies a function is defined.

    Parameters
    ----------
    source_code : str
        Python source code to validate

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    Validate valid code::

        >>> errors = validate_py_source("def process(item, index, state, **ports): return item")
        >>> assert errors == []

    Validate invalid syntax::

        >>> errors = validate_py_source("def process( invalid")
        >>> assert "Syntax error" in errors[0]

    Validate missing function::

        >>> errors = validate_py_source("x = 1")
        >>> assert "must define a function" in errors[0]
    """
    errors: list[str] = []

    if not source_code or not source_code.strip():
        errors.append("!py block is empty. Must define a function.")
        return errors

    # Check syntax by compiling
    try:
        compiled = compile(source_code, "<yaml-!py>", "exec")
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return errors
    except Exception as e:
        errors.append(f"Compilation error: {e}")
        return errors

    # Execute to check for function definition
    namespace: dict[str, Any] = {}
    try:
        exec(compiled, namespace)  # noqa: S102
    except Exception as e:
        errors.append(f"Execution error: {e}")
        return errors

    # Check that a function is defined
    has_function = False
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if callable(obj) and not isinstance(obj, type):
            has_function = True
            break

    if not has_function:
        defined = [k for k in namespace if not k.startswith("_")]
        errors.append(
            f"!py block must define a function. Found: {defined if defined else 'nothing'}"
        )

    return errors


def register_py_tag() -> None:
    """Register the !py custom tag with YAML SafeLoader.

    This function should be called during module initialization to enable
    !py tag support in YAML parsing.

    Examples
    --------
    Import the module to auto-register::

        import hexdag.core.pipeline_builder.py_tag  # Registers !py tag

    Or explicitly register::

        from hexdag.core.pipeline_builder.py_tag import register_py_tag
        register_py_tag()
    """
    # Check if already registered to avoid duplicate registration
    if "!py" not in yaml.SafeLoader.yaml_constructors:
        yaml.SafeLoader.add_constructor("!py", py_constructor)
        logger.debug("Registered !py YAML tag")


# Auto-register when module is imported
register_py_tag()
