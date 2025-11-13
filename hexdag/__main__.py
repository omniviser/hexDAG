"""Entry point for running hexDAG as a module.

Supports both CLI and MCP server modes:
- python -m hexdag [CLI command]
- python -m hexdag --mcp (starts MCP server)
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main entry point for module execution."""
    # Check if --mcp flag is present
    if "--mcp" in sys.argv:
        # Remove --mcp flag and run MCP server
        sys.argv.remove("--mcp")
        from hexdag.mcp_server import mcp

        mcp.run()
    else:
        # Run regular CLI
        from hexdag.cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()
