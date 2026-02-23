#!/usr/bin/env python
"""hexDAG System Libraries Demo: Order Processing Lifecycle.

Demonstrates how hexDAG's system libraries (libs) work together:
- ProcessRegistry: tracks pipeline runs like ``ps`` in Linux
- EntityState: manages order state machine (new -> paid -> shipped -> delivered)
- Scheduler: schedules follow-up pipeline runs (delayed / recurring)

Libs are the hexDAG equivalent of Linux shared libraries (libc, libm).
Every public async method on a HexDAGLib subclass auto-becomes an agent tool.

Run:
    uv run python examples/libs/run_order_lifecycle.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hexdag.compiler.yaml_builder import YamlPipelineBuilder
from hexdag.drivers.observer_manager.local import LocalObserverManager
from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.kernel.lib_base import get_lib_tool_schemas
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.stdlib.adapters.mock import MockLLM
from hexdag.stdlib.lib import (
    EntityState,
    ProcessRegistry,
    ProcessRegistryObserver,
    Scheduler,
)
from hexdag.stdlib.lib.entity_state import InvalidTransitionError
from hexdag.stdlib.lib.process_registry_observer import PROCESS_REGISTRY_EVENTS

console = Console()


# ── 1. Setup libs ────────────────────────────────────────────────────


def setup_libs() -> tuple[ProcessRegistry, EntityState, Scheduler]:
    """Instantiate and configure all three system libs."""

    # ProcessRegistry — tracks pipeline runs
    registry = ProcessRegistry()

    # EntityState — order state machine
    entity_state = EntityState()
    entity_state.register_machine(
        StateMachineConfig(
            entity_type="order",
            states={"new", "paid", "shipped", "delivered", "cancelled"},
            initial_state="new",
            transitions={
                "new": {"paid", "cancelled"},
                "paid": {"shipped", "cancelled"},
                "shipped": {"delivered"},
            },
        )
    )

    # Scheduler — delayed/recurring pipeline execution (no spawner in demo)
    scheduler = Scheduler(spawner=None)

    return registry, entity_state, scheduler


def show_tool_discovery(
    registry: ProcessRegistry,
    entity_state: EntityState,
    scheduler: Scheduler,
) -> None:
    """Display the tools auto-discovered from each lib."""
    table = Table(title="Auto-Discovered Lib Tools (via HexDAGLib.get_tools())")
    table.add_column("Lib", style="cyan", width=20)
    table.add_column("Tool Name", style="green")
    table.add_column("Description", style="dim")

    for lib_name, lib in [
        ("ProcessRegistry", registry),
        ("EntityState", entity_state),
        ("Scheduler", scheduler),
    ]:
        schemas = get_lib_tool_schemas(lib)
        for schema in schemas:
            fn = schema["function"]
            table.add_row(lib_name, fn["name"], fn["description"])

    console.print(table)
    console.print()


# ── 2. Run pipeline with ProcessRegistry observer ────────────────────


async def run_pipeline_with_observer(registry: ProcessRegistry) -> None:
    """Load and run a YAML pipeline, tracking the run in ProcessRegistry."""
    console.print(
        Panel(
            "Loading [bold]order_processing.yaml[/bold] and running through orchestrator.\n"
            "The ProcessRegistryObserver auto-populates the registry from pipeline events.",
            title="Step 2: Run Pipeline",
        )
    )

    # Wire observer
    observer_mgr = LocalObserverManager()
    observer = ProcessRegistryObserver(registry)
    observer_mgr.register(
        observer,
        event_types=PROCESS_REGISTRY_EVENTS,
        keep_alive=True,
    )

    # Build pipeline from YAML
    yaml_path = str(Path(__file__).parent / "order_processing.yaml")
    builder = YamlPipelineBuilder()
    graph, _config = builder.build_from_yaml_file(yaml_path)

    # Create orchestrator with mock LLM + observer
    mock_llm = MockLLM(
        responses=[
            '{"approved": true, "reason": "Valid payment method"}',
            '{"status": "shipped", "tracking_id": "TRK-987654"}',
        ],
        delay_seconds=0.05,
    )
    orchestrator = Orchestrator(
        ports={"llm": mock_llm, "observer_manager": observer_mgr},
    )

    # Run the pipeline
    console.print("  Running pipeline...", style="dim")
    results = await orchestrator.run(
        graph,
        initial_input={
            "order_id": "ORD-001",
            "customer_id": "CUST-42",
            "amount": "299.99",
        },
    )
    console.print("  Pipeline completed.\n", style="green")

    # Query the registry
    runs = await registry.alist()
    if runs:
        table = Table(title="ProcessRegistry: Pipeline Runs (via alist())")
        table.add_column("Run ID", style="cyan", max_width=36)
        table.add_column("Pipeline", style="green")
        table.add_column("Status", style="bold")
        table.add_column("Duration (ms)")
        for run in runs:
            table.add_row(
                run["run_id"],
                run["pipeline_name"],
                run["status"],
                f"{run.get('duration_ms', 'N/A')}",
            )
        console.print(table)
    else:
        console.print("  [yellow]No runs recorded (observer may not have fired).[/yellow]")

    console.print()
    return results


# ── 3. EntityState demo ──────────────────────────────────────────────


async def demo_entity_state(entity_state: EntityState) -> None:
    """Walk an order through the state machine, including an invalid transition."""
    console.print(
        Panel(
            "State machine: [bold]new -> paid -> shipped -> delivered[/bold]\n"
            "Also shows what happens on an invalid transition.",
            title="Step 3: EntityState — Order State Machine",
        )
    )

    order_id = "ORD-001"

    # Register entity
    result = await entity_state.aregister_entity("order", order_id)
    console.print(f"  Registered: {result}")

    # Valid transitions
    transitions = [
        ("paid", "Payment received"),
        ("shipped", "Handed to carrier"),
        ("delivered", "Customer confirmed receipt"),
    ]
    for to_state, reason in transitions:
        result = await entity_state.atransition("order", order_id, to_state, reason=reason)
        console.print(f"  Transition: {result['from_state']} -> {result['to_state']}")

    # Invalid transition (delivered -> paid)
    console.print()
    console.print("  Attempting invalid transition: delivered -> paid", style="yellow")
    try:
        await entity_state.atransition("order", order_id, "paid")
    except InvalidTransitionError as exc:
        console.print(f"  Caught InvalidTransitionError: {exc}", style="red")

    # Show current state
    console.print()
    state = await entity_state.aget_state("order", order_id)
    console.print(f"  Current state: {state}")

    # Show history
    history = await entity_state.aget_history("order", order_id)
    table = Table(title="EntityState: Audit History (via aget_history())")
    table.add_column("From", style="dim")
    table.add_column("To", style="green")
    table.add_column("Reason", style="cyan")
    for entry in history:
        reason = entry.get("metadata", {}).get("reason", "")
        table.add_row(
            str(entry["from_state"] or "(initial)"),
            entry["to_state"],
            reason,
        )
    console.print(table)
    console.print()


# ── 4. Scheduler demo ───────────────────────────────────────────────


async def demo_scheduler(scheduler: Scheduler) -> None:
    """Schedule tasks, list them, and cancel one."""
    console.print(
        Panel(
            "Schedule delayed and recurring pipeline runs.\n"
            "No spawner configured — scheduler logs instead of executing.",
            title="Step 4: Scheduler — Delayed & Recurring Execution",
        )
    )

    # Schedule a one-shot follow-up
    once_result = await scheduler.aschedule_once(
        pipeline_name="order-follow-up",
        initial_input={"order_id": "ORD-001"},
        delay_seconds=3600.0,
        ref_id="ORD-001",
        ref_type="order",
    )
    console.print(
        f"  Scheduled one-shot: {once_result['task_id'][:8]}... "
        f"(delay={once_result['delay_seconds']}s)"
    )

    # Schedule a recurring health check
    recurring_result = await scheduler.aschedule_recurring(
        pipeline_name="system-health-check",
        interval_seconds=300.0,
        ref_id="system",
        ref_type="infra",
    )
    console.print(
        f"  Scheduled recurring: {recurring_result['task_id'][:8]}... "
        f"(interval={recurring_result['interval_seconds']}s)"
    )

    # List all tasks
    tasks = await scheduler.alist_scheduled()
    table = Table(title="Scheduler: Scheduled Tasks (via alist_scheduled())")
    table.add_column("Task ID", style="cyan", max_width=12)
    table.add_column("Pipeline", style="green")
    table.add_column("Type")
    table.add_column("Status", style="bold")
    table.add_column("Ref ID", style="dim")
    for task in tasks:
        table.add_row(
            task["task_id"][:12] + "...",
            task["pipeline_name"],
            task["schedule_type"],
            task["status"],
            task.get("ref_id", ""),
        )
    console.print(table)

    # Cancel the recurring task
    cancel_result = await scheduler.acancel(recurring_result["task_id"])
    console.print(f"\n  Cancelled recurring task: cancelled={cancel_result['cancelled']}")

    # Cleanup
    await scheduler.ateardown()
    console.print()


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]hexDAG System Libraries Demo[/bold cyan]\n"
            "Order Processing Lifecycle with ProcessRegistry, EntityState, and Scheduler",
            border_style="bold",
        )
    )
    console.print()

    # Step 1: Setup
    console.print(
        Panel(
            "Instantiate libs and register the order state machine.\n"
            "Each lib's public async methods are auto-exposed as agent tools.",
            title="Step 1: Setup Libs",
        )
    )
    registry, entity_state, scheduler = setup_libs()
    show_tool_discovery(registry, entity_state, scheduler)

    # Step 2: Run pipeline
    await run_pipeline_with_observer(registry)

    # Step 3: EntityState
    await demo_entity_state(entity_state)

    # Step 4: Scheduler
    await demo_scheduler(scheduler)

    # Done
    console.print(
        Panel.fit(
            "[green]All 3 system libs demonstrated successfully.[/green]\n"
            "ProcessRegistry tracked pipeline runs, EntityState enforced the order lifecycle,\n"
            "and Scheduler managed delayed/recurring tasks.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/yellow]")
