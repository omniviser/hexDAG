"""Runs commands for HexDAG CLI - monolithic in-memory executor."""

import threading
import time
import uuid
from typing import Any

import click
import typer
from rich.console import Console
from rich.table import Table

from hexdag.cli.commands.run_cmd import run_pipeline
from hexdag.cli.utils import print_output

app = typer.Typer()
console = Console()

# Global in-memory store for runs
RUN_STORE: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()


def _execute_run(
    run_id: str, pipeline_path: str, input_file: str | None, max_concurrency: int, timeout: int
) -> None:
    """Internal executor function for a run."""
    with RUN_LOCK:
        RUN_STORE[run_id]["status"] = "running"
        RUN_STORE[run_id]["logs"] = []
        RUN_STORE[run_id]["start_time"] = time.time()

    def log(msg: str) -> None:
        with RUN_LOCK:
            RUN_STORE[run_id]["logs"].append(msg)
        console.print(msg)

    try:
        # Run pipeline using the monolithic executor
        run_pipeline(
            pipeline_path=pipeline_path,
            input_file=input_file,
            validate=True,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )
        with RUN_LOCK:
            RUN_STORE[run_id]["status"] = "completed"
            RUN_STORE[run_id]["end_time"] = time.time()
    except Exception as e:
        with RUN_LOCK:
            RUN_STORE[run_id]["status"] = "failed"
            RUN_STORE[run_id]["end_time"] = time.time()
            RUN_STORE[run_id]["logs"].append(str(e))
        log(f"[red]Run {run_id} failed: {e}[/red]")


@app.command("run")
def start_run(
    pipeline_path: str = typer.Argument(..., help="Pipeline YAML path"),
    input_file: str = typer.Option(None, "--input", "-i", help="JSON input file for pipeline"),
    max_concurrency: int = typer.Option(1, "--max-concurrency"),
    timeout: int = typer.Option(60, "--timeout"),
) -> None:
    """Start a new run."""
    run_id = str(uuid.uuid4())
    with RUN_LOCK:
        RUN_STORE[run_id] = {
            "pipeline": pipeline_path,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "logs": [],
        }
    console.print(f"[cyan]Run started:[/cyan] {run_id}")
    # Execute in a separate thread
    t = threading.Thread(
        target=_execute_run, args=(run_id, pipeline_path, input_file, max_concurrency, timeout)
    )
    t.start()


@app.command("list")
def list_runs(
    limit: int = typer.Option(50, "--limit"), since: str | None = typer.Option(None)
) -> None:
    """List previous runs."""
    ctx = click.get_current_context()
    if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
        with RUN_LOCK:
            serial = [{"run_id": rid, **run} for rid, run in list(RUN_STORE.items())[:limit]]
        print_output(serial, ctx)
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Run ID")
    table.add_column("Pipeline")
    table.add_column("Status")
    table.add_column("Start Time")
    table.add_column("End Time")
    with RUN_LOCK:
        for run_id, run in list(RUN_STORE.items())[:limit]:
            table.add_row(
                run_id,
                run["pipeline"],
                run["status"],
                str(run["start_time"]),
                str(run["end_time"]),
            )
    console.print(table)


@app.command("status")
def run_status(run_id: str) -> None:
    """Show status of a run."""
    with RUN_LOCK:
        run = RUN_STORE.get(run_id)
        if not run:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise typer.Exit(1)
        ctx = click.get_current_context()
        if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
            print_output({"run_id": run_id, "status": run["status"]}, ctx)
        else:
            console.print(f"[bold]Run {run_id}[/bold]: {run['status']}")


@app.command("cancel")
def run_cancel(run_id: str) -> None:
    """Cancel a running run (thread stop simulated)."""
    with RUN_LOCK:
        run = RUN_STORE.get(run_id)
        if not run:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise typer.Exit(1)
        if run["status"] != "running":
            console.print(f"[yellow]Run {run_id} not running[/yellow]")
            raise typer.Exit(0)
        run["status"] = "cancelled"
    console.print(f"[cyan]Run {run_id} marked as cancelled[/cyan]")


@app.command("resume")
def run_resume(run_id: str) -> None:
    """Resume a cancelled run."""
    with RUN_LOCK:
        run = RUN_STORE.get(run_id)
        if not run:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise typer.Exit(1)
        if run["status"] not in ["cancelled", "failed"]:
            console.print(f"[yellow]Run {run_id} cannot be resumed[/yellow]")
            raise typer.Exit(0)
        pipeline_path = run["pipeline"]
    console.print(f"[cyan]Resuming run {run_id}[/cyan]")
    t = threading.Thread(target=_execute_run, args=(run_id, pipeline_path, None, 1, 60))
    t.start()


@app.command("results")
def run_results(run_id: str, out: str | None = typer.Option(None, "--out")) -> None:
    """Show results of a run (logs only in monolith)."""
    with RUN_LOCK:
        run = RUN_STORE.get(run_id)
        if not run:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise typer.Exit(1)
        ctx = click.get_current_context()
        if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
            print_output({"run_id": run_id, "logs": run["logs"]}, ctx)
        else:
            console.print(f"[bold]Results / Logs for {run_id}[/bold]")
            for log_line in run["logs"]:
                console.print(log_line)
        if out:
            import json

            with open(out, "w") as f:
                json.dump(run["logs"], f, indent=2)
            console.print(f"[dim]Logs saved to {out}[/dim]")


@app.command("logs")
def run_logs(run_id: str, follow: bool = typer.Option(False, "--follow")) -> None:
    """Tail logs of a run."""
    import time

    last_idx = 0
    while True:
        with RUN_LOCK:
            run = RUN_STORE.get(run_id)
            if not run:
                console.print(f"[red]Run {run_id} not found[/red]")
                raise typer.Exit(1)
            logs = run["logs"]
        new_logs = logs[last_idx:]
        for line in new_logs:
            console.print(line)
        last_idx += len(new_logs)
        if not follow:
            break
        time.sleep(0.5)
