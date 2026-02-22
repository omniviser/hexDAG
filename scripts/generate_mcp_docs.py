#!/usr/bin/env python3
"""Generate MCP documentation from code artifacts.

This script uses hexDAG's autodiscovery system to find all components and
generates documentation for the MCP server guide functions. Generated docs
are auto-extracted from actual code (signatures, docstrings, schemas) to
stay up-to-date.

Usage:
    uv run python scripts/generate_mcp_docs.py

Generated docs are saved to docs/generated/mcp/ and should NOT be manually edited.
Pre-commit hooks can be configured to regenerate on source changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

from hexdag.api import components
from hexdag.docs.extractors import DocExtractor
from hexdag.docs.generators import GuideGenerator
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import resolve

if TYPE_CHECKING:
    from hexdag.docs.models import AdapterDoc, NodeDoc, ToolDoc

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs" / "generated" / "mcp"


def collect_adapters() -> list[AdapterDoc]:
    """Discover and extract documentation from all adapters.

    Uses ``api.components.list_adapters()`` for autodiscovery, then
    resolves each class and extracts full documentation via DocExtractor.

    Returns
    -------
    list[AdapterDoc]
        List of adapter documentation objects
    """
    adapters = []

    for adapter_info in components.list_adapters():
        module_path = adapter_info["module_path"]
        try:
            cls = resolve(module_path)
            adapter_doc = DocExtractor.extract_adapter_doc(cls)
            adapters.append(adapter_doc)
            logger.debug(f"Extracted adapter: {cls.__name__}")
        except Exception as e:
            logger.warning(f"Failed to extract adapter {module_path}: {e}")

    return adapters


def collect_nodes() -> list[NodeDoc]:
    """Discover and extract documentation from all nodes.

    Uses ``api.components.list_nodes()`` for autodiscovery, then
    resolves each class and extracts full documentation via DocExtractor.

    Returns
    -------
    list[NodeDoc]
        List of node documentation objects
    """
    nodes = []

    for node_info in components.list_nodes():
        module_path = node_info["module_path"]
        try:
            cls = resolve(module_path)
            node_doc = DocExtractor.extract_node_doc(cls)
            nodes.append(node_doc)
            logger.debug(f"Extracted node: {cls.__name__}")
        except Exception as e:
            logger.warning(f"Failed to extract node {module_path}: {e}")

    return nodes


def collect_tools() -> list[ToolDoc]:
    """Discover and extract documentation from all tools.

    Uses ``api.components.list_tools()`` for autodiscovery, then
    resolves each function and extracts full documentation via DocExtractor.

    Returns
    -------
    list[ToolDoc]
        List of tool documentation objects
    """
    tools = []

    for tool_info in components.list_tools():
        module_path = tool_info["module_path"]
        try:
            # Tools are functions, not classes — resolve() only handles classes.
            # Split "module.func_name" and import the attribute directly.
            mod_path, _, attr_name = module_path.rpartition(".")
            mod = importlib.import_module(mod_path)
            func = getattr(mod, attr_name)
            tool_doc = DocExtractor.extract_tool_doc(func)
            tools.append(tool_doc)
            logger.debug(f"Extracted tool: {func.__name__}")
        except Exception as e:
            logger.warning(f"Failed to extract tool {module_path}: {e}")

    return tools


def main() -> int:
    """Generate all MCP documentation files.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error)
    """
    print("Generating MCP documentation from code...")
    print()

    # Ensure output directory exists
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect documentation via autodiscovery
    print("Scanning components...")
    adapters = collect_adapters()
    print(f"  Found {len(adapters)} adapters")

    nodes = collect_nodes()
    print(f"  Found {len(nodes)} nodes")

    tools = collect_tools()
    print(f"  Found {len(tools)} tools")
    print()

    # Initialize generator
    generator = GuideGenerator()

    def write_with_newline(path: Path, content: str) -> None:
        """Write content ensuring trailing newline."""
        if not content.endswith("\n"):
            content += "\n"
        path.write_text(content)

    # Generate adapter guide
    try:
        adapter_guide = generator.generate_adapter_guide(adapters)
        output_path = DOCS_DIR / "adapter_guide.md"
        write_with_newline(output_path, adapter_guide)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating adapter guide: {e}", file=sys.stderr)
        return 1

    # Generate node guide
    try:
        node_guide = generator.generate_node_guide(nodes)
        output_path = DOCS_DIR / "node_guide.md"
        write_with_newline(output_path, node_guide)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating node guide: {e}", file=sys.stderr)
        return 1

    # Generate tool guide
    try:
        tool_guide = generator.generate_tool_guide(tools)
        output_path = DOCS_DIR / "tool_guide.md"
        write_with_newline(output_path, tool_guide)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating tool guide: {e}", file=sys.stderr)
        return 1

    # Generate syntax reference
    try:
        syntax_reference = generator.generate_syntax_reference()
        output_path = DOCS_DIR / "syntax_reference.md"
        write_with_newline(output_path, syntax_reference)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating syntax reference: {e}", file=sys.stderr)
        return 1

    # Generate extension guide
    try:
        extension_guide = generator.generate_extension_guide(adapters, nodes, tools)
        output_path = DOCS_DIR / "extension_guide.md"
        write_with_newline(output_path, extension_guide)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating extension guide: {e}", file=sys.stderr)
        return 1

    # Generate pipeline schema guide
    try:
        schema_guide = generator.generate_pipeline_schema_guide()
        output_path = DOCS_DIR / "pipeline_schema_guide.md"
        write_with_newline(output_path, schema_guide)
        print(f"Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating pipeline schema guide: {e}", file=sys.stderr)
        return 1

    print()
    print("All MCP documentation generated successfully!")
    print()
    print(f"Generated files in: {DOCS_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
