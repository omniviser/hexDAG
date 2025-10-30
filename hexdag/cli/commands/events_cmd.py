"""Events CLI commands for HexDAG."""

import json
import time
from datetime import datetime, timedelta
from typing import TypedDict, cast

import click
import typer
from rich.console import Console
from rich.table import Table

from hexdag.cli.utils import print_output

"""Events CLI commands for HexDAG."""


app = typer.Typer()
console = Console()


class Event(TypedDict):
    id: str
    timestamp: datetime
    namespace: str
    action: str
    pipeline: str
    run_id: str
    status: str
    payload: dict


EVENTS_DB: list[Event] = [
    {
        "id": "evt1",
        "timestamp": datetime.now() - timedelta(minutes=10),
        "namespace": "node",
        "action": "completed",
        "pipeline": "pipeline1",
        "run_id": "run1",
        "status": "success",
        "payload": {"foo": 123},
    },
    {
        "id": "evt2",
        "timestamp": datetime.now() - timedelta(minutes=5),
        "namespace": "pipeline",
        "action": "failed",
        "pipeline": "pipeline1",
        "run_id": "run1",
        "status": "error",
        "payload": {"error": "Something went wrong"},
    },
]


@app.command("tail")
def tail(
    pipeline: str | None = typer.Option(None, help="Pipeline to filter"),
    run: str | None = typer.Option(None, help="Run ID to filter"),
    namespace: str | None = typer.Option("*", help="Namespace filter"),
    filter: str | None = typer.Option(None, help="Filter by namespace:status"),
    follow: bool = typer.Option(False, help="Follow new events"),
    json_out: bool = typer.Option(False, "--json", help="Output in JSON"),
) -> None:
    """Stream events."""

    def matches(event: Event) -> bool:
        if pipeline and event["pipeline"] != pipeline:
            return False
        if run and event["run_id"] != run:
            return False
        if namespace != "*" and event["namespace"] != namespace:
            return False
        if filter:
            for f in filter.split(","):
                ns, status = f.split(":")
                if event["namespace"] == ns and event["status"] != status:
                    return False
        return True

    last_seen = 0
    while True:
        for idx, evt in enumerate(EVENTS_DB[last_seen:], start=last_seen):
            if matches(evt):
                ctx = click.get_current_context()
                if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
                    print_output(evt, ctx)
                elif json_out:
                    console.print(json.dumps(evt, default=str))
                else:
                    console.print(
                        f"[{evt['namespace']}] {evt['action']} - {evt['status']} "
                        f"({evt['pipeline']}/{evt['run_id']})"
                    )

            last_seen = idx + 1
        if not follow:
            break
        time.sleep(1)


@app.command("list")
def list_events(
    pipeline: str | None = typer.Option(None),
    run: str | None = typer.Option(None),
    namespace: str = typer.Option("*"),
    since: str = typer.Option("24h"),
    limit: int = typer.Option(200),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List historical events."""
    now = datetime.now()
    delta = timedelta(hours=24)
    if since.endswith("m"):
        delta = timedelta(minutes=int(since[:-1]))
    elif since.endswith("h"):
        delta = timedelta(hours=int(since[:-1]))

    filtered = []
    for evt in EVENTS_DB:
        if evt["timestamp"] < now - delta:
            continue
        if pipeline and evt["pipeline"] != pipeline:
            continue
        if run and evt["run_id"] != run:
            continue
        if namespace != "*" and evt["namespace"] != namespace:
            continue
        filtered.append(evt)
        if len(filtered) >= limit:
            break

    ctx = click.get_current_context()
    if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
        print_output(filtered, ctx)
        return

    if json_out:
        console.print(json.dumps(filtered, default=str))
    else:
        table = Table(title="Events")
        table.add_column("ID", style="cyan")
        table.add_column("Namespace", style="yellow")
        table.add_column("Action", style="green")
        table.add_column("Status", style="red")
        table.add_column("Pipeline", style="magenta")
        table.add_column("Run", style="white")
        table.add_column("Timestamp", style="blue")
        for evt in filtered:
            table.add_row(
                evt["id"],
                evt["namespace"],
                evt["action"],
                evt["status"],
                evt["pipeline"],
                evt["run_id"],
                str(evt["timestamp"]),
            )
        console.print(table)


@app.command("describe")
def describe(event_id: str, validate: bool = typer.Option(False)) -> None:
    """Show detailed event info."""
    for evt in EVENTS_DB:
        if evt["id"] == event_id:
            console.print_json(data=evt) if validate else console.print(evt)
            return
    console.print(f"[red]Event {event_id} not found[/red]")


@app.command("stats")
def stats(
    by: str = typer.Option("pipeline", help="Group by pipeline, namespace, or severity"),
    since: str = typer.Option("24h"),
    errors_only: bool = typer.Option(False),
) -> None:
    """Show event statistics."""
    now = datetime.now()
    delta = timedelta(hours=24)
    if since.endswith("m"):
        delta = timedelta(minutes=int(since[:-1]))
    elif since.endswith("h"):
        delta = timedelta(hours=int(since[:-1]))

    counts: dict[str, int] = {}
    for evt in EVENTS_DB:
        if evt["timestamp"] < now - delta:
            continue
        if errors_only and evt["status"] != "error":
            continue
        key = cast("str", evt.get(by, evt["namespace"]))
        counts[key] = counts.get(key, 0) + 1

    table = Table(title="Event Stats")
    table.add_column(by.capitalize(), style="cyan")
    table.add_column("Count", style="green")
    for k, v in counts.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command("export")
def export(
    pipeline: str | None = typer.Option(None),
    run: str | None = typer.Option(None),
    namespace: str | None = typer.Option("*"),
    format: str = typer.Option("json"),
    output: str = typer.Option(...),
) -> None:
    """Export events to a file."""
    filtered = [
        evt
        for evt in EVENTS_DB
        if (pipeline is None or evt["pipeline"] == pipeline)
        and (run is None or evt["run_id"] == run)
        and (namespace == "*" or evt["namespace"] == namespace)
    ]

    if format == "json":
        with open(output, "w") as f:
            json.dump(filtered, f, default=str)
    elif format == "csv":
        import csv

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
            writer.writeheader()
            writer.writerows(filtered)
    else:
        console.print(f"[red]Unsupported format: {format}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Exported {len(filtered)} events to {output}[/green]")
