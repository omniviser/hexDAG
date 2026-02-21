#!/usr/bin/env python3
"""Generate MCP documentation from code artifacts.

This script scans hexDAG builtin components and generates documentation
for the MCP server guide functions. Generated docs are auto-extracted from
actual code (signatures, docstrings, schemas) to stay up-to-date.

Usage:
    uv run python scripts/generate_mcp_docs.py

Generated docs are saved to docs/generated/mcp/ and should NOT be manually edited.
Pre-commit hooks can be configured to regenerate on source changes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hexdag.core.docs.extractors import DocExtractor
from hexdag.core.docs.generators import GuideGenerator
from hexdag.core.logging import get_logger

if TYPE_CHECKING:
    from hexdag.core.docs.models import AdapterDoc, NodeDoc, ToolDoc

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs" / "generated" / "mcp"


def collect_adapters() -> list[AdapterDoc]:
    """Scan and extract documentation from all builtin adapters.

    Returns
    -------
    list[AdapterDoc]
        List of adapter documentation objects
    """
    adapters = []

    # Adapter modules to scan
    adapter_modules = [
        "hexdag.builtin.adapters.openai.openai_adapter",
        "hexdag.builtin.adapters.anthropic.anthropic_adapter",
        "hexdag.builtin.adapters.memory.in_memory_memory",
        "hexdag.builtin.adapters.memory.sqlite_memory_adapter",
        "hexdag.builtin.adapters.memory.file_memory_adapter",
        "hexdag.builtin.adapters.memory.session_memory",
        "hexdag.builtin.adapters.memory.state_memory",
        "hexdag.builtin.adapters.database.sqlite.sqlite_adapter",
        "hexdag.builtin.adapters.database.csv.csv_adapter",
        "hexdag.builtin.adapters.mock.mock_llm",
        "hexdag.builtin.adapters.mock.mock_database",
        "hexdag.builtin.adapters.mock.mock_tool_router",
        "hexdag.builtin.adapters.mock.mock_embedding",
        "hexdag.builtin.adapters.secret.local_secret_adapter",
        "hexdag.builtin.adapters.local.local_observer_manager",
        "hexdag.builtin.adapters.local.local_policy_manager",
        "hexdag.builtin.adapters.unified_tool_router",
    ]

    for module_path in adapter_modules:
        try:
            module = importlib.import_module(module_path)

            # Find adapter classes in module
            for name in dir(module):
                if name.startswith("_"):
                    continue

                obj = getattr(module, name, None)
                if obj is None or not isinstance(obj, type):
                    continue

                # Check if it looks like an adapter
                if not name.endswith(("Adapter", "Memory", "Router", "Manager")):
                    continue

                # Skip base classes
                if name in ("BaseAdapter", "ConfigurableAdapter"):
                    continue

                try:
                    adapter_doc = DocExtractor.extract_adapter_doc(obj)
                    adapters.append(adapter_doc)
                    logger.debug(f"Extracted adapter: {name}")
                except Exception as e:
                    logger.warning(f"Failed to extract adapter {name}: {e}")

        except ImportError as e:
            logger.warning(f"Could not import {module_path}: {e}")

    return adapters


def collect_nodes() -> list[NodeDoc]:
    """Scan and extract documentation from all builtin nodes.

    Returns
    -------
    list[NodeDoc]
        List of node documentation objects
    """
    nodes = []

    try:
        from hexdag.builtin import nodes as builtin_nodes

        for name in dir(builtin_nodes):
            if name.startswith("_"):
                continue

            obj = getattr(builtin_nodes, name, None)
            if obj is None or not isinstance(obj, type):
                continue

            # Check if it's a node class
            if not name.endswith("Node"):
                continue

            # Skip base classes
            if name in ("BaseNodeFactory",):
                continue

            try:
                node_doc = DocExtractor.extract_node_doc(obj)
                nodes.append(node_doc)
                logger.debug(f"Extracted node: {name}")
            except Exception as e:
                logger.warning(f"Failed to extract node {name}: {e}")

    except ImportError as e:
        logger.warning(f"Could not import builtin nodes: {e}")

    return nodes


def collect_tools() -> list[ToolDoc]:
    """Scan and extract documentation from all builtin tools.

    Returns
    -------
    list[ToolDoc]
        List of tool documentation objects
    """
    tools = []

    try:
        from hexdag.core.domain import agent_tools

        for name in dir(agent_tools):
            if name.startswith("_"):
                continue

            obj = getattr(agent_tools, name, None)
            if obj is None:
                continue

            # Check if it's a callable (function)
            if not callable(obj):
                continue

            # Skip non-tool items
            if name in ("Any", "TypeVar", "TYPE_CHECKING"):
                continue

            # Skip classes
            if isinstance(obj, type):
                continue

            try:
                tool_doc = DocExtractor.extract_tool_doc(obj)
                tools.append(tool_doc)
                logger.debug(f"Extracted tool: {name}")
            except Exception as e:
                logger.warning(f"Failed to extract tool {name}: {e}")

    except ImportError as e:
        logger.warning(f"Could not import builtin tools: {e}")

    # Also try database tools
    try:
        from hexdag.builtin.tools import database_tools

        for name in dir(database_tools):
            if name.startswith("_"):
                continue

            obj = getattr(database_tools, name, None)
            if obj is None or not callable(obj) or isinstance(obj, type):
                continue

            try:
                tool_doc = DocExtractor.extract_tool_doc(obj)
                tools.append(tool_doc)
                logger.debug(f"Extracted tool: {name}")
            except Exception as e:
                logger.warning(f"Failed to extract tool {name}: {e}")

    except ImportError:
        pass  # database_tools may not exist

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

    # Collect documentation from code
    print("Scanning components...")
    adapters = collect_adapters()
    print(f"  → Found {len(adapters)} adapters")

    nodes = collect_nodes()
    print(f"  → Found {len(nodes)} nodes")

    tools = collect_tools()
    print(f"  → Found {len(tools)} tools")
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
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating adapter guide: {e}", file=sys.stderr)
        return 1

    # Generate node guide
    try:
        node_guide = generator.generate_node_guide(nodes)
        output_path = DOCS_DIR / "node_guide.md"
        write_with_newline(output_path, node_guide)
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating node guide: {e}", file=sys.stderr)
        return 1

    # Generate tool guide
    try:
        tool_guide = generator.generate_tool_guide(tools)
        output_path = DOCS_DIR / "tool_guide.md"
        write_with_newline(output_path, tool_guide)
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating tool guide: {e}", file=sys.stderr)
        return 1

    # Generate syntax reference
    try:
        syntax_reference = generator.generate_syntax_reference()
        output_path = DOCS_DIR / "syntax_reference.md"
        write_with_newline(output_path, syntax_reference)
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating syntax reference: {e}", file=sys.stderr)
        return 1

    # Generate extension guide
    try:
        extension_guide = generator.generate_extension_guide(adapters, nodes, tools)
        output_path = DOCS_DIR / "extension_guide.md"
        write_with_newline(output_path, extension_guide)
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating extension guide: {e}", file=sys.stderr)
        return 1

    # Generate pipeline schema guide
    try:
        schema_guide = generator.generate_pipeline_schema_guide()
        output_path = DOCS_DIR / "pipeline_schema_guide.md"
        write_with_newline(output_path, schema_guide)
        print(f"✓ Generated {output_path.relative_to(Path.cwd())}")
    except Exception as e:
        print(f"Error generating pipeline schema guide: {e}", file=sys.stderr)
        return 1

    print()
    print("✓ All MCP documentation generated successfully!")
    print()
    print(f"Generated files in: {DOCS_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
