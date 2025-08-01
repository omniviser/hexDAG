"""Command-line interface for hexAI pipelines and development tools."""


# Avoid module-level imports to prevent warnings when using python -m
def main() -> None:
    """Entry point for CLI."""
    from .simple_pipeline_cli import main as cli_main

    return cli_main()


__all__ = ["main"]
