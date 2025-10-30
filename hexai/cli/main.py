"""Compatibility shim: delegate CLI entrypoint to `hexdag.cli.main`.

This file used to be the main CLI implementation under the old package
name (`hexai`). Keep a small shim here so imports referencing
`hexai.cli.main` continue to work while the canonical implementation
lives in `hexdag.cli.main`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def main(argv: list[str] | None = None) -> Any:
    """Delegate to hexdag.cli.main.main()."""
    mod = import_module("hexdag.cli.main")
    # Call the main() if available, otherwise expose module for direct usage
    return getattr(mod, "main", lambda *a, **k: None)(argv)


if __name__ == "__main__":
    main()
