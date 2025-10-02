"""Plugin development commands for hexDAG CLI."""

import subprocess  # nosec B404 - subprocess is used safely with controlled inputs
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer()
console = Console()


def get_plugin_dir() -> Path:
    """Get the hexai_plugins directory path."""
    # Try to find the project root
    current = Path.cwd()
    while current != current.parent:
        if (current / "hexai_plugins").exists():
            return current / "hexai_plugins"
        if (current / "pyproject.toml").exists():
            # Check if this is the hexdag project
            with open(current / "pyproject.toml") as f:
                if "hexdag" in f.read():
                    plugin_dir = current / "hexai_plugins"
                    if not plugin_dir.exists():
                        plugin_dir.mkdir(parents=True, exist_ok=True)
                    return plugin_dir
        current = current.parent

    # Default to current directory
    plugin_dir = Path.cwd() / "hexai_plugins"
    if not plugin_dir.exists():
        console.print("[yellow]Creating hexai_plugins directory in current location[/yellow]")
        plugin_dir.mkdir(parents=True, exist_ok=True)
    return plugin_dir


@app.command("new")
def create_plugin(
    name: Annotated[str, typer.Argument(help="Name of the new plugin (e.g., redis_adapter)")],
    port: Annotated[str, typer.Option("--port", "-p", help="Port type to implement")] = "database",
    author: Annotated[
        str, typer.Option("--author", "-a", help="Plugin author name")
    ] = "HexDAG Team",
) -> None:
    """Create a new plugin from template."""
    plugin_dir = get_plugin_dir()
    plugin_path = plugin_dir / name

    if plugin_path.exists():
        console.print(f"[red]Error: Plugin '{name}' already exists at {plugin_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Creating new plugin: {name}[/green]")

    # Create directory structure
    plugin_path.mkdir(parents=True)
    (plugin_path / "tests").mkdir()

    # Create __init__.py
    class_name = name.replace("_", " ").title().replace(" ", "")
    init_content = f'''"""${name} plugin for hexDAG."""

from .{name} import {class_name}

__all__ = ["{class_name}"]
'''
    (plugin_path / "__init__.py").write_text(init_content)

    # Create main adapter file
    class_name = name.replace("_", " ").title().replace(" ", "")
    adapter_content = f'''"""{class_name} implementation."""

from typing import Any

from hexai.core.registry.decorators import adapter


@adapter(
    name="{name.replace("_adapter", "")}",
    implements_port="{port}",
    namespace="plugin",
    description="{class_name} for hexDAG",
)
class {class_name}:
    """{class_name} implementation.

    This adapter implements the {port} port interface.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize {name}."""
        # Initialize your adapter here
        pass

    # TODO: Implement the {port} port interface methods
    # Check hexai/core/ports/{port}.py for the interface definition

    def __repr__(self) -> str:
        """String representation."""
        return f"{class_name}()"
'''
    (plugin_path / f"{name}.py").write_text(adapter_content)

    # Create pyproject.toml
    pyproject_content = f"""[project]
name = "hexdag-{name.replace("_", "-")}"
version = "0.1.0"
description = "{class_name} plugin for hexDAG"
authors = [{{ name = "{author}" }}]
requires-python = "~=3.12.0"
readme = "README.md"
license = "MIT"
keywords = ["hexdag", "plugin", "{port}", "adapter"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    # Add your plugin-specific dependencies here
    # Do not add hexdag as a dependency (it's the parent project)
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hexdag.plugin]
name = "{name.replace("_adapter", "")}"
module = "hexai_plugins.{name}"
port = "{port}"
description = "{class_name} for hexDAG"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
warn_return_any = true
disallow_untyped_defs = false
check_untyped_defs = true
"""
    (plugin_path / "pyproject.toml").write_text(pyproject_content)

    # Create README
    readme_content = f"""# {class_name}

A hexDAG plugin that provides {name.replace("_", " ")} functionality.

## Installation

```bash
# From the hexDAG root directory
pip install -e hexai_plugins/{name}/
```

## Configuration

Add to your hexDAG configuration:

```toml
# In hexdag.toml or pyproject.toml
[tool.hexdag]
plugins = [
    "hexai_plugins.{name}",
]
```

## Usage

The {class_name} implements the `{port}` port interface.

```python
from hexai.core.registry import registry

# After bootstrap
adapter = registry.get("{name.replace("_adapter", "")}", namespace="plugin")
```

## Development

```bash
# Run tests
hexdag plugin test {name}

# Format code
hexdag plugin format {name}

# Lint code
hexdag plugin lint {name}
```
"""
    (plugin_path / "README.md").write_text(readme_content)

    # Create test file
    test_content = f'''"""Tests for {name}."""

import pytest


class Test{class_name}:
    """Test suite for {class_name}."""

    def test_adapter_initialization(self):
        """{class_name} should initialize without errors."""
        from hexai_plugins.{name}.{name} import {class_name}

        adapter = {class_name}()
        assert adapter is not None
        assert repr(adapter) == "{class_name}()"

    # TODO: Add more tests for your adapter functionality
'''
    (plugin_path / "tests" / f"test_{name}.py").write_text(test_content)

    # Create LICENSE
    license_content = """MIT License

Copyright (c) 2024 HexDAG Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    (plugin_path / "LICENSE").write_text(license_content)

    # Display success message with next steps
    console.print(Panel(f"[green]✓ Plugin '{name}' created successfully![/green]"))

    table = Table(title="Next Steps", show_header=False, box=None)
    table.add_row("1.", f"Edit [cyan]{plugin_path / f'{name}.py'}[/cyan] to implement your adapter")
    table.add_row(
        "2.", f"Update the port interface methods based on [cyan]hexai/core/ports/{port}.py[/cyan]"
    )
    table.add_row("3.", f"Add dependencies to [cyan]{plugin_path / 'pyproject.toml'}[/cyan]")
    table.add_row("4.", f"Run [yellow]hexdag plugin lint {name}[/yellow] to check your code")
    table.add_row(
        "5.", f"Run [yellow]hexdag plugin test {name}[/yellow] to test your implementation"
    )
    console.print(table)


@app.command("list")
def list_plugins() -> None:
    """List all available plugins."""
    plugin_dir = get_plugin_dir()

    if not plugin_dir.exists():
        console.print("[yellow]No plugins directory found[/yellow]")
        return

    plugins = [
        d
        for d in plugin_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ]

    if not plugins:
        console.print("[yellow]No plugins found[/yellow]")
        return

    table = Table(title="Available Plugins", show_header=True)
    table.add_column("Plugin", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Description")

    for plugin_path in plugins:
        name = plugin_path.name
        pyproject_path = plugin_path / "pyproject.toml"

        if pyproject_path.exists():
            import tomllib

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                version = data.get("project", {}).get("version", "unknown")
                plugin_info = data.get("tool", {}).get("hexdag", {}).get("plugin", {})
                port = plugin_info.get("port", "unknown")
                description = plugin_info.get("description", "No description")
        else:
            version = "unknown"
            port = "unknown"
            description = "No pyproject.toml found"

        table.add_row(name, version, port, description)

    console.print(table)


@app.command("lint")
def lint_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to lint")],
    fix: Annotated[bool, typer.Option("--fix", "-f", help="Auto-fix issues")] = True,
) -> None:
    """Lint a plugin's code."""
    plugin_dir = get_plugin_dir()
    plugin_path = plugin_dir / name

    if not plugin_path.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Linting {name}...[/yellow]")

    # Run ruff check
    check_cmd = ["ruff", "check", str(plugin_path)]
    if fix:
        check_cmd.append("--fix")

    result = subprocess.run(check_cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(f"[red]{result.stderr}[/red]")

    # Run ruff format
    format_cmd = ["ruff", "format", str(plugin_path)]
    result = subprocess.run(format_cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        console.print(result.stdout)

    if result.returncode == 0:
        console.print(f"[green]✓ Plugin '{name}' linted successfully[/green]")
    else:
        console.print(f"[red]✗ Linting failed for '{name}'[/red]")
        raise typer.Exit(1)


@app.command("format")
def format_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to format")],
) -> None:
    """Format a plugin's code."""
    plugin_dir = get_plugin_dir()
    plugin_path = plugin_dir / name

    if not plugin_path.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Formatting {name}...[/yellow]")

    # Run ruff format
    cmd = ["ruff", "format", str(plugin_path), "--line-length=100"]
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603

    if result.stdout:
        console.print(result.stdout)

    if result.returncode == 0:
        console.print(f"[green]✓ Plugin '{name}' formatted successfully[/green]")
    else:
        console.print(f"[red]✗ Formatting failed for '{name}'[/red]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
        raise typer.Exit(1)


@app.command("test")
def test_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to test")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Run tests for a plugin."""
    plugin_dir = get_plugin_dir()
    plugin_path = plugin_dir / name

    if not plugin_path.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        raise typer.Exit(1)

    test_dir = plugin_path / "tests"
    if not test_dir.exists():
        console.print(f"[yellow]No tests found for '{name}'[/yellow]")
        return

    console.print(f"[yellow]Testing {name}...[/yellow]")

    # Run pytest
    cmd = [sys.executable, "-m", "pytest", str(test_dir)]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603

    if result.stdout:
        console.print(result.stdout)

    if result.returncode == 0:
        console.print(f"[green]✓ All tests passed for '{name}'[/green]")
    else:
        console.print(f"[red]✗ Tests failed for '{name}'[/red]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
        raise typer.Exit(1)


@app.command("install")
def install_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to install")],
    editable: Annotated[
        bool, typer.Option("--editable", "-e", help="Install in editable mode")
    ] = True,
) -> None:
    """Install a plugin in development mode."""
    plugin_dir = get_plugin_dir()
    plugin_path = plugin_dir / name

    if not plugin_path.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Installing {name}...[/yellow]")

    # Install with pip
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(str(plugin_path))

    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603

    if result.stdout:
        console.print(result.stdout)

    if result.returncode == 0:
        console.print(f"[green]✓ Plugin '{name}' installed successfully[/green]")
    else:
        console.print(f"[red]✗ Installation failed for '{name}'[/red]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
        raise typer.Exit(1)


@app.command("check-all")
def check_all_plugins() -> None:
    """Run lint and test for all plugins."""
    plugin_dir = get_plugin_dir()

    if not plugin_dir.exists():
        console.print("[yellow]No plugins directory found[/yellow]")
        return

    plugins = [
        d.name
        for d in plugin_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ]

    if not plugins:
        console.print("[yellow]No plugins found[/yellow]")
        return

    console.print(f"[bold]Checking {len(plugins)} plugins...[/bold]\n")

    results = []
    for plugin_name in plugins:
        console.print(f"[cyan]Checking {plugin_name}...[/cyan]")

        # Lint
        lint_success = True
        try:
            lint_plugin(plugin_name, fix=True)
        except typer.Exit:
            lint_success = False

        # Test
        test_success = True
        try:
            test_plugin(plugin_name)
        except typer.Exit:
            test_success = False

        results.append((plugin_name, lint_success, test_success))
        console.print()  # Add spacing

    # Summary
    console.print("[bold]Summary:[/bold]")
    table = Table(show_header=True)
    table.add_column("Plugin", style="cyan")
    table.add_column("Lint", style="green")
    table.add_column("Test", style="green")

    for plugin_name, lint_ok, test_ok in results:
        lint_status = "[green]✓[/green]" if lint_ok else "[red]✗[/red]"
        test_status = "[green]✓[/green]" if test_ok else "[red]✗[/red]"
        table.add_row(plugin_name, lint_status, test_status)

    console.print(table)

    all_passed = all(lint_ok and test_ok for _, lint_ok, test_ok in results)
    if all_passed:
        console.print("\n[green]✓ All plugins passed checks![/green]")
    else:
        console.print("\n[red]✗ Some plugins failed checks[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
