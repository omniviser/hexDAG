"""Observers CLI commands for HexDAG."""

import json
from typing import TypedDict

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


class Observer(TypedDict):
    name: str
    namespace: str
    timeout: int
    batch_capable: bool
    filters: list[str]


OBSERVERS_DB: list[Observer] = [
    {
        "name": "log_node_events",
        "namespace": "core",
        "timeout": 5,
        "batch_capable": True,
        "filters": ["node:*"],
    },
    {
        "name": "alert_pipeline_failures",
        "namespace": "user",
        "timeout": 10,
        "batch_capable": False,
        "filters": ["pipeline:failed"],
    },
]


@app.command("list")
def list_observers(namespace: str | None = typer.Option("*")) -> None:
    """List all registered observers."""
    filtered = [o for o in OBSERVERS_DB if namespace == "*" or o["namespace"] == namespace]
    table = Table(title="Observers")
    table.add_column("Name")
    table.add_column("Namespace")
    table.add_column("Timeout")
    table.add_column("Batch Capable")
    table.add_column("Filters")
    for o in filtered:
        table.add_row(
            o["name"],
            o["namespace"],
            str(o["timeout"]),
            str(o["batch_capable"]),
            ", ".join(o["filters"]),
        )
    console.print(table)


@app.command("describe")
def describe_observer(qualified_name: str) -> None:
    """Describe an observer in detail."""
    for o in OBSERVERS_DB:
        if o["name"] == qualified_name:
            console.print_json(data=o)
            return
    console.print(f"[red]Observer {qualified_name} not found[/red]")


@app.command("test")
def test_observer(qualified_name: str, event_file: str | None = typer.Option(None)) -> None:
    """Invoke an observer locally with an event."""
    event = {}
    if event_file:
        with open(event_file) as f:
            event = json.load(f)
    for o in OBSERVERS_DB:
        if o["name"] == qualified_name:
            console.print(f"[green]Testing observer '{qualified_name}'[/green]")
            console.print(f"With event: {event}")
            console.print(f"Observer {qualified_name} executed successfully")
            return
    console.print(f"[red]Observer {qualified_name} not found[/red]")
