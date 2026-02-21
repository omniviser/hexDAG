"""Jinja2 template rendering plugin — Phase 2 of the rendering pipeline.

Build-time rendering for metadata and config; runtime preservation for
node specs and pipeline outputs. Templates inside node ``spec`` fields
(where ``kind != "Pipeline"``) and ``outputs`` fields are **skipped** to
preserve them for Phase 3 runtime rendering via ``PromptTemplate``.

Error messages are prefixed with ``[Phase 2: Build-Time Template Rendering]``.

See Also
--------
- ``preprocessing/env_vars.py`` — Phase 1: ``${VAR}`` resolution
- ``core/orchestration/prompt/template.py`` — Phase 3: runtime rendering
"""

from __future__ import annotations

from contextlib import suppress
from functools import singledispatch
from typing import Any

from jinja2 import TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from hexdag.core.pipeline_builder.preprocessing._type_guards import _is_dict_config


class TemplatePlugin:
    """Render Jinja2 templates in YAML with two-phase rendering strategy.

    **Build-time Rendering** (YAML configuration context):
    - Metadata fields (e.g., node names, descriptions)
    - Pipeline-level spec fields (e.g., variables, ports, policies)
    - Enables dynamic configuration from environment/variables

    **Runtime Rendering** (node execution context):
    - Node spec fields (e.g., template, prompt_template, initial_prompt)
    - Pipeline outputs (e.g., spec.outputs with {{node.result}} references)
    - Preserved to allow access to dependency/node outputs at runtime
    - Enables dynamic prompts and output mapping based on execution results

    Example:
        ```yaml
        spec:
          variables:
            node_name: analyzer
          nodes:
            - kind: llm_node
              metadata:
                name: "{{ spec.variables.node_name }}"  # Build-time: renders to "analyzer"
              spec:
                template: "{{input}}"  # Runtime: renders when node executes
          outputs:
            result: "{{analyzer.analysis}}"  # Runtime: rendered after pipeline completes
        ```

    Security:
        Uses SandboxedEnvironment to prevent arbitrary code execution.
    """

    def __init__(self) -> None:
        # Use sandboxed environment to prevent code execution attacks
        # Even though this is for config files, defense in depth is important
        self.env = SandboxedEnvironment(autoescape=False, keep_trailing_newline=True)

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Render Jinja2 templates with config as context."""
        result = _render_templates(config, config, self.env)
        if not _is_dict_config(result):
            raise TypeError(
                f"Template rendering must return a dictionary, got {type(result).__name__}"
            )
        return result


@singledispatch
def _render_templates(obj: Any, context: dict[str, Any], env: Any) -> Any:
    """Recursively render Jinja2 templates.

    Returns:
        - For primitives: Returns the primitive unchanged
        - For strings: Returns str | int | float | bool (with type coercion)
        - For dicts: Returns dict[str, Any]
        - For lists: Returns list[Any]
    """
    return obj


@_render_templates.register(str)
def _render_templates_str(obj: str, context: dict[str, Any], env: Any) -> str | int | float | bool:
    """Render Jinja2 template in string."""
    from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilderError

    if "{{" not in obj and "{%" not in obj:
        return obj

    try:
        rendered: str = env.from_string(obj).render(context)
        # Type coercion
        with suppress(ValueError):
            return int(rendered)
        with suppress(ValueError):
            return float(rendered)
        if rendered.lower() in ("true", "yes"):
            return True
        if rendered.lower() in ("false", "no"):
            return False
        return rendered
    except TemplateSyntaxError as e:
        raise YamlPipelineBuilderError(
            f"[Phase 2: Build-Time Template Rendering] "
            f"Invalid Jinja2 template syntax: {e}\n"
            f"  Template: {obj}\n"
            f"  Hint: If this template should render at runtime with node outputs, "
            f"place it inside a node's 'spec' field where it is automatically preserved."
        ) from e
    except UndefinedError as e:
        raise YamlPipelineBuilderError(
            f"[Phase 2: Build-Time Template Rendering] "
            f"Undefined variable in template: {e}\n"
            f"  Template: {obj}\n"
            f"  Hint: Build-time variables must be defined in the YAML config "
            f"(e.g., under 'spec.variables'). If this references node outputs, "
            f"move it inside a node's 'spec' field for runtime rendering."
        ) from e


@_render_templates.register(dict)
def _render_templates_dict(obj: dict, context: dict[str, Any], env: Any) -> dict[str, Any]:
    """Render templates in dict values.

    Skip rendering for node spec fields and pipeline outputs to avoid conflicts between
    YAML-level templating and runtime template strings.

    Strategy:
    - Node specs: Preserve for runtime rendering with dependency outputs
    - Pipeline outputs: Preserve for runtime rendering with node results
    - Metadata and config: Render at build time with YAML context
    """
    result = {}
    for k, v in obj.items():
        # Check if this is a node spec (not a Pipeline spec) by looking for the 'kind' sibling key
        # Node kinds end with '_node' (e.g., 'prompt_node', 'llm_node', 'function_node')
        # Pipeline kind is 'Pipeline' - we should NOT skip its spec
        if (
            k == "spec"
            and isinstance(v, dict)
            and "kind" in obj
            and isinstance(obj.get("kind"), str)
            and obj["kind"] != "Pipeline"
        ):
            # This is a node spec - preserve template strings for runtime rendering
            result[k] = v  # Preserve entire spec as-is
        # Also preserve 'outputs' field in Pipeline spec - references node results
        elif k == "outputs" and isinstance(v, dict):
            # Pipeline outputs reference node results (e.g., {{node.output}})
            # Preserve for runtime rendering after nodes execute
            result[k] = v
        else:
            result[k] = _render_templates(v, context, env)
    return result


@_render_templates.register(list)
def _render_templates_list(obj: list, context: dict[str, Any], env: Any) -> list[Any]:
    """Render templates in list items."""
    return [_render_templates(item, context, env) for item in obj]
