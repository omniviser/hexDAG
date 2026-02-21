"""Environment variable resolution plugin — Phase 1 of the rendering pipeline.

Resolves ``${VAR}`` and ``${VAR:default}`` patterns at build time.
Secret-like variables (``*_API_KEY``, ``*_SECRET``, ``*_TOKEN``,
``*_PASSWORD``, ``*_CREDENTIAL``) are deferred to Phase 3b runtime
resolution in ``component_instantiator._resolve_deferred_env_vars()``.

Error messages are prefixed with ``[Phase 1: Environment Variable Resolution]``.

See Also
--------
- ``preprocessing/template.py`` — Phase 2: Jinja2 rendering
- ``component_instantiator.py`` — Phase 3b: deferred resolution
"""

from __future__ import annotations

import os
import re
from contextlib import suppress
from functools import singledispatch
from typing import Any

from hexdag.core.logging import get_logger
from hexdag.core.pipeline_builder.preprocessing._type_guards import _is_dict_config

logger = get_logger(__name__)


class EnvironmentVariablePlugin:
    """Resolve ${VAR} and ${VAR:default} in YAML with deferred secret resolution.

    For KeyVault/SecretPort workflows, secret-like environment variables are
    preserved as ${VAR} for runtime resolution. This allows:
    - Building pipelines without secrets present
    - Runtime secret injection via SecretPort -> Memory
    - Separation of build and deployment contexts

    Secret patterns (deferred to runtime):
    - *_API_KEY, *_SECRET, *_TOKEN, *_PASSWORD, *_CREDENTIAL
    - SECRET_*

    Non-secret variables are resolved immediately at build-time.
    """

    ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*?)(?::([^}]*))?\}")

    # Secret patterns that should be deferred to runtime
    SECRET_PATTERNS = frozenset({
        r".*_API_KEY$",
        r".*_SECRET$",
        r".*_TOKEN$",
        r".*_PASSWORD$",
        r".*_CREDENTIAL$",
        r"^SECRET_.*",
    })

    def __init__(self, defer_secrets: bool = True):
        """Initialize environment variable plugin.

        Parameters
        ----------
        defer_secrets : bool, default=True
            If True, preserve ${VAR} syntax for secret-like variables,
            allowing runtime resolution from KeyVault/Memory.
            If False, all variables are resolved at build-time (legacy behavior).
        """
        self.defer_secrets = defer_secrets
        if defer_secrets:
            # Compile secret detection regex
            self._secret_regex: re.Pattern[str] | None = re.compile(
                "|".join(f"({p})" for p in self.SECRET_PATTERNS)
            )
        else:
            self._secret_regex = None

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve environment variables."""
        result = _resolve_env_vars(
            config,
            self.ENV_VAR_PATTERN,
            secret_regex=self._secret_regex,
            defer_secrets=self.defer_secrets,
        )
        if not _is_dict_config(result):
            raise TypeError(
                f"Environment variable resolution must return a dictionary, "
                f"got {type(result).__name__}"
            )
        return result


@singledispatch
def _resolve_env_vars(
    obj: Any,
    pattern: re.Pattern[str],
    secret_regex: re.Pattern[str] | None = None,
    defer_secrets: bool = True,
) -> Any:
    """Recursively resolve ${VAR} in any structure.

    Parameters
    ----------
    obj : Any
        Object to process
    pattern : re.Pattern[str]
        Regex pattern for matching ${VAR} syntax
    secret_regex : re.Pattern[str] | None
        Regex for detecting secret-like variable names
    defer_secrets : bool
        If True, preserve ${VAR} for secrets

    Returns
    -------
    Any
        - For primitives: Returns the primitive unchanged
        - For strings: Returns str | int | float | bool (with type coercion)
        - For dicts: Returns dict[str, Any]
        - For lists: Returns list[Any]
    """
    return obj


@_resolve_env_vars.register(str)
def _resolve_env_vars_str(
    obj: str,
    pattern: re.Pattern[str],
    secret_regex: re.Pattern[str] | None = None,
    defer_secrets: bool = True,
) -> str | int | float | bool:
    """Resolve ${VAR} in strings with optional secret deferral."""
    from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilderError

    def replacer(match: re.Match[str]) -> str:
        var_name, default = match.group(1), match.group(2)

        # Check if this looks like a secret
        if defer_secrets and secret_regex and secret_regex.match(var_name):
            # Secret detected - preserve ${VAR} syntax for runtime resolution
            logger.debug(f"Deferring secret variable to runtime: {var_name}")
            return match.group(0)  # Return original ${VAR} or ${VAR:default}

        # Non-secret - resolve immediately from environment
        env_value = os.environ.get(var_name)
        if env_value is None:
            if default is not None:
                return default
            raise YamlPipelineBuilderError(
                f"[Phase 1: Environment Variable Resolution] "
                f"Environment variable '${{{var_name}}}' is not set and has no default.\n"
                f"  In: {obj}\n"
                f"  Hint: Use ${{{var_name}:default_value}} to provide a default, "
                f"or set the variable before building the pipeline."
            )
        return env_value

    resolved = pattern.sub(replacer, obj)

    # Type coercion only if the value changed (was resolved)
    if resolved != obj and not (defer_secrets and resolved.startswith("${")):
        if resolved.lower() in ("true", "yes", "1"):
            return True
        if resolved.lower() in ("false", "no", "0"):
            return False
        with suppress(ValueError):
            return int(resolved)
        with suppress(ValueError):
            return float(resolved)
    return resolved


@_resolve_env_vars.register(dict)
def _resolve_env_vars_dict(
    obj: dict,
    pattern: re.Pattern[str],
    secret_regex: re.Pattern[str] | None = None,
    defer_secrets: bool = True,
) -> dict[str, Any]:
    """Resolve ${VAR} in dict values."""
    return {k: _resolve_env_vars(v, pattern, secret_regex, defer_secrets) for k, v in obj.items()}


@_resolve_env_vars.register(list)
def _resolve_env_vars_list(
    obj: list,
    pattern: re.Pattern[str],
    secret_regex: re.Pattern[str] | None = None,
    defer_secrets: bool = True,
) -> list[Any]:
    """Resolve ${VAR} in list items."""
    return [_resolve_env_vars(item, pattern, secret_regex, defer_secrets) for item in obj]
