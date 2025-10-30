"""Policies CLI commands for HexDAG."""

import json
from typing import TypedDict

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


class Policy(TypedDict):
    name: str
    namespace: str
    priority: int
    filters: list[str]
    description: str


POLICIES_DB: list[Policy] = [
    {
        "name": "node_retry_policy",
        "namespace": "core",
        "priority": 10,
        "filters": ["node:failed"],
        "description": "Retries failed nodes automatically",
    },
    {
        "name": "pipeline_timeout_policy",
        "namespace": "user",
        "priority": 5,
        "filters": ["pipeline:timeout"],
        "description": "Fails pipelines exceeding timeout",
    },
]


@app.command("list")
def list_policies(namespace: str | None = typer.Option("*")) -> None:
    """List all registered policies."""
    filtered = [p for p in POLICIES_DB if namespace == "*" or p["namespace"] == namespace]
    table = Table(title="Policies")
    table.add_column("Name")
    table.add_column("Namespace")
    table.add_column("Priority")
    table.add_column("Filters")
    table.add_column("Description")
    for p in filtered:
        table.add_row(
            p["name"], p["namespace"], str(p["priority"]), ", ".join(p["filters"]), p["description"]
        )
    console.print(table)


@app.command("describe")
def describe_policy(qualified_name: str) -> None:
    """Describe a policy in detail."""
    for p in POLICIES_DB:
        if p["name"] == qualified_name:
            console.print_json(data=p)
            return
    console.print(f"[red]Policy {qualified_name} not found[/red]")


@app.command("simulate")
def simulate_policy(
    pipeline_yaml: str,
    event: str = typer.Option(..., help="Event to simulate, e.g., node:failed"),
    context_file: str | None = typer.Option(None),
) -> None:
    """Dry-run a policy decision without executing pipeline."""
    context = {}
    if context_file:
        with open(context_file) as f:
            context = json.load(f)
    matched = [p for p in POLICIES_DB if event in p["filters"]]
    console.print(f"[green]Simulating event '{event}' on pipeline '{pipeline_yaml}'[/green]")
    if context:
        console.print(f"With context: {context}")
    if matched:
        console.print("Policies triggered:")
        for p in matched:
            console.print(f"- {p['name']} (priority {p['priority']})")
    else:
        console.print("[yellow]No policies triggered[/yellow]")
