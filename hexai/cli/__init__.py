"""Command-line interface for hexAI pipelines and development tools.

Note: The CLI requires the 'cli' extra to be installed:
    pip install hexdag[cli]
"""


# Avoid module-level imports to prevent warnings when using python -m
def main() -> None:
    """Entry point for CLI."""
    try:
        import yaml  # noqa: F401
    except ImportError as e:
        print("Error: PyYAML is not installed.")
        print("Please install with:")
        print("  pip install hexdag[cli]")
        print("  or")
        print("  uv pip install hexdag[cli]")
        raise SystemExit(1) from e

    from hexai.cli.simple_pipeline_cli import main as cli_main

    return cli_main()


__all__ = ["main"]
