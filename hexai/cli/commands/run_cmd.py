"""Run command for HexDAG CLI - monolithic executor."""

import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from hexai.cli.commands.registry_cmd import ComponentType, bootstrap_registry, registry

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer()
console = Console()


def topological_sort(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simple topological sort based on 'depends_on'."""
    sorted_nodes = []
    node_map = {n["name"]: n for n in nodes}
    visited = set()

    def visit(node_name: str) -> None:
        if node_name in visited:
            return
        node = node_map[node_name]
        for dep in node.get("depends_on", []):
            visit(dep)
        visited.add(node_name)
        sorted_nodes.append(node)

    for n in nodes:
        visit(n["name"])

    return sorted_nodes


@app.command("run")
def run_pipeline(
    pipeline_path: str = typer.Argument(..., help="Path to pipeline YAML"),
    input_file: str | None = typer.Option(
        None, "--input", "-i", help="JSON input file for pipeline"
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate pipeline before run"
    ),
    max_concurrency: int = typer.Option(
        1, "--max-concurrency", help="Max parallel nodes (ignored in this monolith)"
    ),
    timeout: int = typer.Option(60, "--timeout", help="Timeout per node in seconds"),
) -> None:
    """Run a pipeline from YAML."""
    # Bootstrap registry
    with contextlib.suppress(Exception):
        bootstrap_registry()

    # Load pipeline YAML
    with open(pipeline_path) as f:
        pipeline = yaml.safe_load(f)

    nodes = pipeline.get("nodes", [])
    if validate:
        if not nodes:
            console.print("[red]Pipeline has no nodes![/red]")
            raise typer.Exit(1)
        for node in nodes:
            if "name" not in node or "func" not in node:
                console.print(f"[red]Node missing 'name' or 'func': {node}[/red]")
                raise typer.Exit(1)

    # Load input overrides
    input_data = {}
    if input_file:
        with open(input_file) as f:
            input_data = json.load(f)

    # Topological sort
    sorted_nodes = topological_sort(nodes)

    results: dict[str, Any] = {}
    console.print(
        f"[bold green]Starting pipeline:[/bold green] {pipeline.get('name', '<unnamed>')}"
    )

    for node in sorted_nodes:
        node_name = node["name"]
        func_name = node["func"]
        params = node.get("params", {})

        # Inject input overrides
        if node_name in input_data:
            params.update(input_data[node_name])

        # Resolve templates {{X.result}}
        for k, v in params.items():
            if isinstance(v, str) and "{{" in v and "}}" in v:
                ref = v.removeprefix("{{").removesuffix("}}").split(".")
                if ref[1] == "result" and ref[0] in results:
                    params[k] = results[ref[0]]

        # Get callable from registry
        comp_info = registry.get_info(func_name)
        if not comp_info or comp_info.component_type != ComponentType.NODE:
            console.print(f"[red]Node function '{func_name}' not found or not a NODE[/red]")
            raise typer.Exit(1)
        func = registry.get(func_name)
        func_callable: Callable[..., Any] = func  # type: ignore

        # Execute
        console.print(f"[cyan]Running node {node_name} -> {func_name}[/cyan]")
        try:
            start = time.time()
            output = func_callable(**params)
            duration = time.time() - start
            results[node_name] = output
            console.print(f"[green]✔ Node {node_name} completed in {duration:.2f}s[/green]")
        except Exception as e:
            console.print(f"[red]✖ Node {node_name} failed: {e}[/red]")
            raise typer.Exit(1) from e

    console.print("\n[bold]Pipeline finished. Results:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Node")
    table.add_column("Result")
    for k, v in results.items():
        table.add_row(k, str(v))
    console.print(table)

    # Optionally save results
    out_file = pipeline.get("results_out")
    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Results saved to {out_file}[/dim]")
