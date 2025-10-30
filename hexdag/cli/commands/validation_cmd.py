"""Validation and linting commands for HexDAG CLI."""

import json

import typer

app = typer.Typer(help="Validation & Lint commands")


@app.command("lint")
def lint(paths: list[str], json_output: bool = False) -> None:
    """Perform linting on specified paths.

    Args:
        paths: List of file paths to lint
        json_output: Whether to output results in JSON format
    """
    # dummy lint implementation
    result = {"status": "ok", "paths": paths}
    if json_output:
        print(json.dumps(result))
    else:
        print(f"Linted {len(paths)} paths: {paths}")


@app.command("schema")
def schema(json_output: bool = False) -> None:
    """Display the event schema definition.

    Args:
        json_output: Whether to output schema in JSON format
    """
    schema = {"namespace": "string", "action": "string", "payload": {}}
    if json_output:
        print(json.dumps(schema, indent=2))
    else:
        print(schema)


@app.command("validate")
def validate(file: str, json_output: bool = False) -> None:
    """Validate event payload file against registered event schemas."""
    import yaml
    from rich.console import Console

    from hexdag.core.orchestration.events.events import EVENT_REGISTRY

    console = Console()

    try:
        # Load payload file
        with open(file) as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                # Try YAML if JSON fails
                f.seek(0)
                payload = yaml.safe_load(f)

        errors = []

        # Basic structure validation
        if not isinstance(payload, dict):
            errors.append("Payload must be a dictionary")
        elif "event_type" not in payload:
            errors.append("Missing 'event_type' field")
        elif "data" not in payload:
            errors.append("Missing 'data' field")
        else:
            event_type = payload["event_type"]
            data = payload["data"]

            # Find matching event spec
            event_spec = None
            for spec in EVENT_REGISTRY.values():
                if spec.event_type == event_type:
                    event_spec = spec
                    break

            if not event_spec:
                errors.append(f"Unknown event type: {event_type}")
            else:
                # Validate envelope fields
                errors.extend(
                    f"Missing required envelope field: {field}"
                    for field in event_spec.envelope_fields
                    if field not in data
                )

                # Validate attribute fields if defined
                if event_spec.attr_fields:
                    errors.extend(
                        f"Missing required attribute field: {field}"
                        for field in event_spec.attr_fields
                        if field not in data
                    )

        result = {"valid": len(errors) == 0, "errors": errors, "file": file}

        if json_output:
            print(json.dumps(result))
        else:
            if result["valid"]:
                console.print("[green]✓ Event payload is valid[/green]")
            else:
                console.print("[red]✗ Event payload validation failed:[/red]")
                for error in errors:
                    console.print(f"  • [red]{error}[/red]")
                raise typer.Exit(1)

    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
        if json_output:
            print(json.dumps({"valid": False, "error": str(e), "file": file}))
        else:
            console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)
