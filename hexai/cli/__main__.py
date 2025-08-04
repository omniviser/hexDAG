#!/usr/bin/env python3
"""Entry point for hexAI CLI tools."""

if __name__ == "__main__":
    # Import only when run as main to avoid circular import warnings
    from hexai.cli.simple_pipeline_cli import main

    main()
