"""MkDocs hooks for generating HexDAG component documentation.

This module provides hooks that integrate with MkDocs to automatically generate
documentation pages from the HexDAG component resolver during the build process.
Documentation is generated from the builtin node factory signatures, making it
adaptive and always up-to-date with the codebase.

Note: This uses the resolver system (hexdag.core.resolver) which provides
module path resolution for components. Components are documented based on
their registered aliases and inspected signatures.
"""
# mypy: ignore-errors
# type: ignore

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.files import File, Files

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

logger = logging.getLogger("mkdocs.hooks.registry_docs")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Generate documentation files from HexDAG resolver.

    This hook runs after the files collection is populated from docs_dir.
    It queries the HexDAG component resolver for registered aliases and
    generates markdown documentation for each component type.

    Args:
        files: Collection of documentation files
        config: MkDocs configuration

    Returns:
        Updated files collection with generated component docs
    """

    try:
        from hexdag.core.resolver import get_builtin_aliases, resolve

        docs_dir = Path(config["docs_dir"])

        # Get all registered builtin aliases
        aliases = get_builtin_aliases()
        if not aliases:
            logger.info("No builtin aliases registered, skipping documentation generation")
            return files

        # Group aliases by component type (inferred from path)
        nodes: dict[str, str] = {}
        adapters: dict[str, str] = {}
        other: dict[str, str] = {}

        for alias, full_path in aliases.items():
            if ".nodes." in full_path or alias.endswith("_node"):
                nodes[alias] = full_path
            elif ".adapters." in full_path or "Adapter" in full_path:
                adapters[alias] = full_path
            else:
                other[alias] = full_path

        # Generate node documentation
        if nodes:
            markdown = _generate_component_docs("Nodes", nodes, resolve)
            file_path = "reference/nodes.md"
            virtual_file = _create_virtual_file(file_path, markdown, docs_dir, config["site_dir"])
            files.append(virtual_file)
            logger.info(f"Generated documentation for {len(nodes)} nodes")

        # Generate adapter documentation
        if adapters:
            markdown = _generate_component_docs("Adapters", adapters, resolve)
            file_path = "reference/adapters.md"
            virtual_file = _create_virtual_file(file_path, markdown, docs_dir, config["site_dir"])
            files.append(virtual_file)
            logger.info(f"Generated documentation for {len(adapters)} adapters")

        # Generate other components documentation
        if other:
            markdown = _generate_component_docs("Other Components", other, resolve)
            file_path = "reference/other.md"
            virtual_file = _create_virtual_file(file_path, markdown, docs_dir, config["site_dir"])
            files.append(virtual_file)
            logger.info(f"Generated documentation for {len(other)} other components")

    except ImportError as e:
        logger.warning(f"Could not import HexDAG resolver: {e}")
    except Exception as e:
        logger.error(f"Error generating component docs: {e}")

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


def _generate_component_docs(title: str, components: dict[str, str], resolve_fn) -> str:
    """Generate markdown documentation for a set of components.

    Args:
        title: Section title (e.g., "Nodes", "Adapters")
        components: Mapping of alias -> full module path
        resolve_fn: Function to resolve module paths to classes

    Returns:
        Markdown content
    """
    lines = [
        f"# {title} Reference",
        "",
        f"This page documents all registered {title.lower()} in HexDAG.",
        "",
        "## Overview",
        "",
        f"Total {title.lower()}: **{len(components)}**",
        "",
        "| Alias | Module Path |",
        "|-------|-------------|",
    ]

    for alias in sorted(components.keys()):
        full_path = components[alias]
        lines.append(f"| `{alias}` | `{full_path}` |")

    lines.extend(["", "---", ""])

    # Generate detailed sections for each component
    for alias in sorted(components.keys()):
        full_path = components[alias]
        try:
            component = resolve_fn(full_path)
            lines.extend(_generate_component_section(alias, full_path, component))
        except Exception as e:
            logger.warning(f"Could not document component {alias}: {e}")
            lines.extend([
                f"### `{alias}`",
                "",
                f"**Module:** `{full_path}`",
                "",
                f"*Could not load component: {e}*",
                "",
                "---",
                "",
            ])

    return "\n".join(lines)


def _generate_component_section(alias: str, full_path: str, component) -> list[str]:
    """Generate markdown section for a single component.

    Args:
        alias: Component alias (e.g., "llm_node")
        full_path: Full module path
        component: The resolved component class

    Returns:
        List of markdown lines
    """
    lines = [
        f"### `{alias}`",
        "",
        f"**Module:** `{full_path}`",
        "",
    ]

    # Add description from docstring
    if hasattr(component, "__doc__") and component.__doc__:
        doc = component.__doc__.strip().split("\n")[0]
        lines.append(doc)
        lines.append("")

    # For nodes (factories), extract __call__ signature
    if callable(component) and callable(component):
        lines.extend(_generate_signature_docs(component))

    # Add YAML usage example for nodes
    if alias.endswith("_node") or "Node" in full_path:
        lines.extend(_generate_yaml_usage(alias, component))

    lines.append("---")
    lines.append("")

    return lines


def _generate_signature_docs(component) -> list[str]:
    """Generate documentation from component's __call__ or __init__ signature.

    Args:
        component: Component class or instance

    Returns:
        List of markdown lines
    """
    lines = []

    try:
        # Try __call__ first (for node factories)
        if callable(component):
            sig = inspect.signature(component.__call__)
        else:
            sig = inspect.signature(component.__init__)

        params = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            type_str = "Any"
            if param.annotation != inspect.Parameter.empty:
                ann = str(param.annotation)
                type_str = (
                    ann.replace("typing.", "")
                    .replace("<class '", "")
                    .replace("'>", "")
                    .replace("hexdag.core.", "")
                )

            # Check if required or optional
            has_default = param.default != inspect.Parameter.empty
            default_str = f" = {param.default!r}" if has_default else ""

            params.append(f"- **`{param_name}`**: `{type_str}`{default_str}")

        if params:
            lines.append("**Parameters:**")
            lines.append("")
            lines.extend(params)
            lines.append("")

    except Exception as e:
        logger.debug(f"Could not extract signature: {e}")

    return lines


def _generate_yaml_usage(alias: str, component) -> list[str]:
    """Generate YAML usage example for a node.

    Args:
        alias: Node alias (e.g., "llm_node")
        component: Node factory class

    Returns:
        List of markdown lines
    """
    lines = [
        "**YAML Usage:**",
        "",
        "```yaml",
    ]

    node_id = alias.replace("_node", "") if alias.endswith("_node") else alias

    # Extract parameters from factory's __call__ signature
    params_list = []

    try:
        if callable(component):
            sig = inspect.signature(component.__call__)

            for param_name, param in sig.parameters.items():
                # Skip self, name, and deps (handled separately in YAML)
                if param_name in ("self", "name", "deps", "kwargs"):
                    continue

                # Check if required
                has_default = param.default != inspect.Parameter.empty
                if not has_default:
                    # Generate example value
                    example_value = _get_example_value(param_name, str(param.annotation))
                    if example_value:
                        params_list.append(f"      {param_name}: {example_value}")

    except Exception as e:
        logger.debug(f"Could not extract signature for {alias}: {e}")

    # Build YAML example
    lines.append("nodes:")
    lines.append(f"  - kind: {alias}")
    lines.append("    metadata:")
    lines.append(f"      name: {node_id}")
    lines.append("    spec:")

    if params_list:
        lines.extend(params_list)
    else:
        lines.append("      # See parameters above")

    lines.append("    dependencies: []")
    lines.append("```")
    lines.append("")

    return lines


def _get_example_value(param_name: str, type_str: str) -> str | None:
    """Get example value for a parameter based on its name and type.

    Args:
        param_name: Parameter name
        type_str: Type string

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
