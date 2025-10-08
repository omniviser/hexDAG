"""Manifest management commands for HexDAG CLI."""

import contextlib
import tomllib
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer()
console = Console()


@app.command("validate")
def validate_manifest(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Path to manifest TOML file"),
    ],
) -> None:
    """Validate manifest file schema and module imports."""

    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest file not found: {manifest_path}[/red]")
        raise typer.Exit(1)

    try:
        # Load and parse TOML
        with open(manifest_path, "rb") as f:
            manifest_data = tomllib.load(f)

        console.print(f"[cyan]Validating manifest: {manifest_path}[/cyan]")

        # Basic structure validation
        issues = []
        warnings = []

        if not isinstance(manifest_data, dict):
            issues.append("Manifest must be a dictionary")  # type: ignore[unreachable]
        else:
            # Check for modules
            if "modules" in manifest_data:
                modules = manifest_data["modules"]
                if not isinstance(modules, list):
                    issues.append("'modules' must be a list")
                else:
                    console.print(f"[dim]Found {len(modules)} module(s)[/dim]")

                    # Try importing each module
                    for i, module in enumerate(modules, 1):
                        try:
                            # Just check if module string is valid
                            if not isinstance(module, str):
                                issues.append(f"Module {i} must be a string")
                            elif not module:
                                issues.append(f"Module {i} is empty")
                            else:
                                # Try to import
                                import importlib

                                importlib.import_module(module)
                                console.print(f"  [green]✓[/green] {module}")
                        except ImportError as e:
                            warnings.append(f"Module '{module}' import warning: {e}")
                            console.print(f"  [yellow]⚠[/yellow] {module} - {e}")

            # Check for plugins
            if "plugins" in manifest_data:
                plugins = manifest_data["plugins"]
                if not isinstance(plugins, list):
                    issues.append("'plugins' must be a list")
                else:
                    console.print(f"[dim]Found {len(plugins)} plugin(s)[/dim]")

        # Report results
        if issues:
            console.print("\n[red]Validation errors:[/red]")
            for issue in issues:
                console.print(f"  [red]✗[/red] {issue}")
            raise typer.Exit(1)

        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  [yellow]⚠[/yellow] {warning}")

        console.print("\n[green]✓ Manifest validation passed[/green]")

    except tomllib.TOMLDecodeError as e:
        console.print(f"[red]TOML parsing error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command("build")
def build_manifest(
    output: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output manifest file path",
        ),
    ] = None,
) -> None:
    """Build manifest from project structure."""
    from hexdag.core.bootstrap import bootstrap_registry
    from hexdag.core.registry import registry

    console.print("[cyan]Building manifest from current registry...[/cyan]")

    # Bootstrap registry
    with contextlib.suppress(Exception):
        bootstrap_registry()

    components = registry.list_components()

    # Group by namespace/module
    modules = set()
    plugins = set()

    for comp in components:
        # Infer module path from component metadata
        if comp.metadata and hasattr(comp.metadata, "raw_component"):
            raw_comp = comp.metadata.raw_component
            if hasattr(raw_comp, "__module__"):
                module = raw_comp.__module__
                # Determine if it's a core module or plugin
                if comp.namespace == "core":
                    modules.add(module.rsplit(".", 1)[0])  # Parent module
                elif comp.namespace == "plugin":
                    plugins.add(module.rsplit(".", 1)[0])

    # Build TOML content
    toml_lines = []
    if modules:
        toml_lines.append("modules = [")
        toml_lines.extend(f'    "{mod}",' for mod in sorted(modules))
        toml_lines.append("]")

    if plugins:
        if toml_lines:
            toml_lines.append("")
        toml_lines.append("plugins = [")
        toml_lines.extend(f'    "{plug}",' for plug in sorted(plugins))
        toml_lines.append("]")

    toml_content = "\n".join(toml_lines)

    # Output
    output_path = output or Path("hexdag-manifest.toml")

    with open(output_path, "w") as f:
        f.write(toml_content)

    console.print(f"[green]✓ Manifest written to: {output_path}[/green]")
    console.print(f"  Modules: {len(modules)}")
    console.print(f"  Plugins: {len(plugins)}")

    # Show preview
    syntax = Syntax(toml_content, "toml", theme="monokai")
    console.print("\n[bold]Preview:[/bold]")
    console.print(syntax)


@app.command("diff")
def diff_manifests(
    manifest_a: Annotated[
        Path,
        typer.Argument(help="First manifest file"),
    ],
    manifest_b: Annotated[
        Path,
        typer.Argument(help="Second manifest file"),
    ],
) -> None:
    """Compare two manifest files and show differences."""
    # Load both manifests
    try:
        with open(manifest_a, "rb") as f:
            data_a = tomllib.load(f)
        with open(manifest_b, "rb") as f:
            data_b = tomllib.load(f)
    except Exception as e:
        console.print(f"[red]Error loading manifests: {e}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Comparing manifests:[/cyan]")
    console.print(f"  A: {manifest_a}")
    console.print(f"  B: {manifest_b}\n")

    # Compare modules
    modules_a = set(data_a.get("modules", []))
    modules_b = set(data_b.get("modules", []))

    added_modules = modules_b - modules_a
    removed_modules = modules_a - modules_b

    # Compare plugins
    plugins_a = set(data_a.get("plugins", []))
    plugins_b = set(data_b.get("plugins", []))

    added_plugins = plugins_b - plugins_a
    removed_plugins = plugins_a - plugins_b

    # Display differences
    has_changes = False

    if added_modules:
        has_changes = True
        console.print("[green]Added modules:[/green]")
        for mod in sorted(added_modules):
            console.print(f"  [green]+[/green] {mod}")

    if removed_modules:
        has_changes = True
        console.print("[red]Removed modules:[/red]")
        for mod in sorted(removed_modules):
            console.print(f"  [red]-[/red] {mod}")

    if added_plugins:
        has_changes = True
        console.print("[green]Added plugins:[/green]")
        for plug in sorted(added_plugins):
            console.print(f"  [green]+[/green] {plug}")

    if removed_plugins:
        has_changes = True
        console.print("[red]Removed plugins:[/red]")
        for plug in sorted(removed_plugins):
            console.print(f"  [red]-[/red] {plug}")

    if not has_changes:
        console.print("[dim]No differences found[/dim]")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Modules: {len(modules_a)} → {len(modules_b)}")
    console.print(f"  Plugins: {len(plugins_a)} → {len(plugins_b)}")
