"""MkDocs hooks for generating HexDAG component registry documentation.

This module provides hooks that integrate with MkDocs to automatically generate
documentation pages from the HexDAG component registry during the build process.
Documentation is generated from the actual node factory signatures, making it
adaptive and always up-to-date with the codebase.
"""
# mypy: ignore-errors
# type: ignore

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.files import File, Files

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

logger = logging.getLogger("mkdocs.hooks.registry_docs")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Generate documentation files from HexDAG registry.

    This hook runs after the files collection is populated from docs_dir.
    It queries the HexDAG component registry and generates markdown files
    for each component type and namespace.

    Supports custom TOML configuration via HEXDAG_CONFIG_PATH environment variable,
    allowing generation of project-specific documentation with custom plugins.

    Args:
        files: Collection of documentation files
        config: MkDocs configuration

    Returns:
        Updated files collection with generated registry docs
    """

    try:
        from hexdag.core.bootstrap import ensure_bootstrapped
        from hexdag.core.registry import registry

        # Get custom config path from environment variable
        custom_config = os.environ.get("HEXDAG_CONFIG_PATH")

        # Ensure registry is bootstrapped with custom config if provided
        if custom_config:
            logger.info(f"Bootstrapping registry with custom config: {custom_config}")
            ensure_bootstrapped(config_path=custom_config)
        else:
            ensure_bootstrapped()

        docs_dir = Path(config["docs_dir"])

        # Generate component type documentation
        component_types = ["node", "adapter", "port", "tool"]

        for comp_type in component_types:
            components = registry.list_components(component_type=comp_type)
            if not components:
                continue

            # Generate markdown content
            markdown = _generate_type_docs(comp_type, components, registry)

            # Create virtual file
            file_path = f"reference/{comp_type}s.md"
            virtual_file = _create_virtual_file(file_path, markdown, docs_dir, config["site_dir"])
            files.append(virtual_file)
            logger.info(f"Generated documentation for {len(components)} {comp_type}s")

        # Generate namespace documentation
        namespaces = _get_all_namespaces(registry)
        for namespace in namespaces:
            markdown = _generate_namespace_docs(namespace, registry)
            file_path = f"namespaces/{namespace}.md"
            virtual_file = _create_virtual_file(file_path, markdown, docs_dir, config["site_dir"])
            files.append(virtual_file)
            logger.info(f"Generated documentation for namespace '{namespace}'")

    except ImportError as e:
        logger.warning(f"Could not import HexDAG registry: {e}")
    except Exception as e:
        logger.error(f"Error generating registry docs: {e}")

    return files


def _create_virtual_file(file_path: str, content: str, docs_dir: Path, site_dir: str) -> File:
    """Create a virtual MkDocs file from generated content.

    Args:
        file_path: Relative path for the generated file
        content: Markdown content
        docs_dir: Documentation source directory
        site_dir: Output site directory

    Returns:
        MkDocs File object
    """
    # Write content to docs_dir so MkDocs can find it
    full_path = docs_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)

    # Create MkDocs File object
    return File(path=file_path, src_dir=str(docs_dir), dest_dir=site_dir, use_directory_urls=True)


def _generate_type_docs(comp_type: str, components: list, registry) -> str:
    """Generate markdown documentation for a component type.

    Args:
        comp_type: Component type (node, adapter, port, tool)
        components: List of ComponentInfo objects
        registry: Component registry instance

    Returns:
        Markdown content
    """
    lines = [
        f"# {comp_type.title()}s Reference",
        "",
        f"This page documents all registered {comp_type}s in HexDAG.",
        "",
        "## Overview",
        "",
        f"Total {comp_type}s: **{len(components)}**",
        "",
    ]

    # Group by namespace
    by_namespace: dict[str, list] = {}
    for comp_info in components:
        by_namespace.setdefault(comp_info.namespace, []).append(comp_info)

    # Generate sections for each namespace
    for namespace in sorted(by_namespace.keys()):
        lines.append(f"## Namespace: `{namespace}`")
        lines.append("")

        for comp_info in sorted(by_namespace[namespace], key=lambda c: c.name):
            try:
                # For ports (abstract classes), use the class directly from metadata
                # For other components, get instance from registry
                if comp_type == "port":
                    component = comp_info.metadata.component.value  # Get the class itself
                else:
                    component = registry.get(comp_info.name, comp_info.namespace)

                lines.extend(
                    _generate_component_section(
                        comp_type, comp_info.namespace, comp_info.name, component, comp_info
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not document {comp_type} {comp_info.namespace}:{comp_info.name}: {e}"
                )

        lines.append("")

    return "\n".join(lines)


def _generate_component_section(
    comp_type: str, namespace: str, name: str, component, comp_info
) -> list[str]:
    """Generate markdown section for a single component.

    Args:
        comp_type: Component type
        namespace: Component namespace
        name: Component name
        component: Component instance (e.g., node factory)
        comp_info: ComponentInfo object with metadata

    Returns:
        List of markdown lines
    """
    lines = [
        f"### `{name}`",
        "",
    ]

    # Add description from metadata
    if comp_info.metadata.description:
        lines.append(comp_info.metadata.description)
        lines.append("")
    elif hasattr(component, "__doc__") and component.__doc__:
        doc = component.__doc__.strip().split("\n")[0]
        lines.append(doc)
        lines.append("")

    # Add metadata
    lines.append("**Metadata:**")
    lines.append("")
    lines.append(f"- **Type:** `{comp_type}`")
    lines.append(f"- **Namespace:** `{namespace}`")
    lines.append(f"- **Name:** `{name}`")
    if comp_info.metadata.subtype:
        lines.append(f"- **Subtype:** `{comp_info.metadata.subtype.value}`")
    lines.append("")

    # Add YAML usage for nodes (generated from factory signature)
    if comp_type == "node" and comp_info.metadata.subtype:
        lines.extend(_generate_yaml_usage_from_signature(comp_info, component))

    # Add schema if available
    if hasattr(component, "get_input_schema"):
        try:
            input_schema = component.get_input_schema()
            if input_schema:
                lines.append("**Input Schema:**")
                lines.append("")
                lines.append("```python")
                lines.append(str(input_schema))
                lines.append("```")
                lines.append("")
        except Exception:
            pass

    if hasattr(component, "get_output_schema"):
        try:
            output_schema = component.get_output_schema()
            if output_schema:
                lines.append("**Output Schema:**")
                lines.append("")
                lines.append("```python")
                lines.append(str(output_schema))
                lines.append("```")
                lines.append("")
        except Exception:
            pass

    lines.append("---")
    lines.append("")

    return lines


def _generate_yaml_usage_from_signature(comp_info, component) -> list[str]:
    """Generate YAML usage from node factory's __call__ signature.

    This makes the documentation adaptive - it reads the actual factory
    signature and generates YAML examples with correct parameters.

    Args:
        comp_info: ComponentInfo object
        component: Component instance (node factory)

    Returns:
        List of markdown lines with YAML examples
    """
    lines = [
        "**YAML Usage:**",
        "",
        "```yaml",
    ]

    subtype = comp_info.metadata.subtype.value
    node_id = comp_info.name.replace("_node", "")

    # Extract parameters from factory's __call__ signature
    params_list = []
    param_docs = []

    try:
        if callable(component):
            sig = inspect.signature(component.__call__)

            for param_name, param in sig.parameters.items():
                # Skip self, name, and deps (handled separately in YAML)
                if param_name in ("self", "name", "deps"):
                    continue

                # Get type annotation
                type_str = "any"
                if param.annotation != inspect.Parameter.empty:
                    ann = str(param.annotation)
                    # Clean up type string
                    type_str = (
                        ann.replace("typing.", "")
                        .replace("<class '", "")
                        .replace("'>", "")
                        .replace("hexai.core.", "")
                    )

                # Check if required or optional
                has_default = param.default != inspect.Parameter.empty
                is_required = not has_default

                # Generate example value based on type and parameter name
                example_value = _get_example_value(
                    param_name, type_str, param.default if has_default else None
                )

                # Format for YAML
                if example_value is not None:
                    comment = (
                        "  # Required"
                        if is_required
                        else f"  # Optional (default: {param.default})"
                    )
                    params_list.append(f"    {param_name}: {example_value}{comment}")

                    # Add to parameter documentation
                    req_str = "**required**" if is_required else "optional"
                    default_str = f", default: `{param.default}`" if has_default else ""
                    param_docs.append(
                        f"- **`{param_name}`** (`{type_str}`, {req_str}{default_str})"
                    )

    except Exception as e:
        logger.debug(f"Could not extract signature for {comp_info.name}: {e}")

    # Build YAML example
    lines.append(f"- type: {subtype}")
    lines.append(f"  id: {node_id}")
    lines.append("  params:")

    if params_list:
        lines.extend(params_list)
    else:
        lines.append("    # No parameters required")

    lines.append("  depends_on: []  # List of upstream node IDs")
    lines.append("```")
    lines.append("")

    # Add parameter documentation
    if param_docs:
        lines.append("**Parameters:**")
        lines.append("")
        lines.extend(param_docs)
        lines.append("")

    return lines


def _get_example_value(param_name: str, type_str: str, default) -> str | None:
    """Get example value for a parameter based on its name and type.

    Args:
        param_name: Parameter name
        type_str: Type string
        default: Default value if any

    Returns:
        Example value as YAML-compatible string
    """
    # Parameter name hints
    if "template" in param_name.lower() or "prompt" in param_name.lower():
        return '"Your prompt here with {{variables}}"'
    if "model" in param_name.lower():
        return '"gpt-4"'
    if "function" in param_name.lower() or "fn" in param_name.lower():
        return '"module.function_name"'
    if "max_steps" in param_name.lower():
        return "10"
    if "max_iterations" in param_name.lower():
        return "100"
    if "temperature" in param_name.lower():
        return "0.7"
    if "max_tokens" in param_name.lower():
        return "1000"
    if "tools" in param_name.lower():
        return "[search, calculator]"
    if "condition" in param_name.lower():
        return '"{{expression}}"'

    # Type-based defaults
    if "str" in type_str.lower():
        return '"value"'
    if "int" in type_str.lower():
        return "0"
    if "float" in type_str.lower():
        return "0.0"
    if "bool" in type_str.lower():
        return "false"
    if "list" in type_str.lower():
        return "[]"
    if "dict" in type_str.lower():
        return "{}"

    return None


def _generate_namespace_docs(namespace: str, registry) -> str:
    """Generate markdown documentation for a namespace.

    Args:
        namespace: Namespace name
        registry: Component registry instance

    Returns:
        Markdown content
    """
    lines = [
        f"# Namespace: {namespace}",
        "",
        f"All components in the `{namespace}` namespace.",
        "",
    ]

    # Get all components in this namespace
    all_components = registry.list_components(namespace=namespace)

    # Group by type
    by_type: dict[str, list[str]] = {}
    for comp_info in all_components:
        comp_type = comp_info.component_type.value.lower()
        by_type.setdefault(comp_type, []).append(comp_info.name)

    # Generate sections
    for comp_type in sorted(by_type.keys()):
        lines.append(f"## {comp_type.title()}s")
        lines.append("")

        lines.extend(
            f"- [`{name}`](../reference/{comp_type}s.md#{name})"
            for name in sorted(by_type[comp_type])
        )

        lines.append("")

    return "\n".join(lines)


def _get_all_namespaces(registry) -> set[str]:
    """Get all unique namespaces from the registry.

    Args:
        registry: Component registry instance

    Returns:
        Set of namespace names
    """
    namespaces = set()
    for comp_info in registry.list_components():
        namespaces.add(comp_info.namespace)
    return namespaces
