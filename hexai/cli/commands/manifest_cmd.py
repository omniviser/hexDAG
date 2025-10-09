"""Manifest management commands for HexDAG CLI."""

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from deepdiff import DeepDiff
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer()
console = Console()


def load_manifest(file_path: Path) -> dict:
    """Load and parse a manifest YAML file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        console.print(f"[red]YAML parsing error in {file_path}: {e}[/red]")
        raise typer.Exit(1) from None
    except FileNotFoundError:
        console.print(f"[red]Manifest file not found: {file_path}[/red]")
        raise typer.Exit(1) from None


@app.command("validate")
def validate(manifest: Path) -> None:
    """
    Validate a manifest file.
    Checks YAML syntax and basic required fields.
    """
    data = load_manifest(manifest)

    errors = []

    if "version" not in data:
        errors.append("Missing required field: 'version'")
    if "components" not in data:
        errors.append("Missing required field: 'components'")
    elif not isinstance(data["components"], list):
        errors.append("'components' must be a list")

    if errors:
        console.print(Panel.fit("\n".join(errors), title="Manifest Errors", border_style="red"))
        raise typer.Exit(1)

    console.print(
        Panel.fit(f"Manifest [green]{manifest}[/green] is valid ✅", border_style="green")
    )


@app.command("build")
def build(
    manifest: Path,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
) -> None:
    """
    Build a manifest into JSON representation.
    Useful for CI/CD or debugging.
    """
    data = load_manifest(manifest)
    json_str = json.dumps(data, indent=2)

    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        console.print(f"[green]Manifest built and written to {output}[/green]")
    else:
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        console.print(syntax)


@app.command("diff")
def diff(old: Path, new: Path) -> None:
    """
    Show differences between two manifests.
    """
    old_data = load_manifest(old)
    new_data = load_manifest(new)

    diff_result = DeepDiff(old_data, new_data, view="tree")

    if not diff_result:
        console.print("[green]No differences found[/green]")
        return

    table = Table(title="Manifest Differences", show_header=True, header_style="bold magenta")
    table.add_column("Change Type")
    table.add_column("Path")
    table.add_column("Details")

    for change_type, changes in diff_result.items():
        for change in changes:
            table.add_row(
                change_type, str(change.path()), str(change.t1) if hasattr(change, "t1") else "-"
            )

    console.print(table)
