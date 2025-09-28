"""Command-line interface for HexDAG.

Note: The CLI requires the 'cli' extra to be installed:
    pip install hexdag[cli]
    or
    uv pip install hexdag[cli]
"""


def main() -> None:
    """Entry point for HexDAG CLI."""
    from hexai.cli.main import main as cli_main

    return cli_main()


__all__ = ["main"]
