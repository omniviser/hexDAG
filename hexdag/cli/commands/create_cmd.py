"""Pipeline creation commands for HexDAG CLI."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

app = typer.Typer()
console = Console()


@app.command("pipeline")
def create_pipeline(
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Pipeline name",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: <name>.yaml)",
        ),
    ] = None,
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template type (minimal, example, full)",
        ),
    ] = "minimal",
) -> None:
    """Create a new pipeline YAML file from a template.

    Examples
    --------
    hexdag create pipeline
    hexdag create pipeline --name my-pipeline
    hexdag create pipeline --template example --output pipeline.yaml
    """
    # Interactive mode if name not provided
    if name is None:
        name = Prompt.ask("[cyan]Pipeline name[/cyan]", default="my-pipeline")

    # Determine output path
    if output is None:
        output = Path(f"{name}.yaml")

    if output.exists():
        if not Confirm.ask(f"[yellow]File {output} already exists. Overwrite?[/yellow]"):
            console.print("[red]Aborted.[/red]")
            raise typer.Exit(1)

    if template == "minimal":
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "TODO: Add description",
            },
            "spec": {
                "nodes": [
                    {
                        "kind": "hexdag.builtin.nodes.FunctionNode",
                        "metadata": {"name": "step1"},
                        "spec": {
                            "fn": "my_module.my_function",
                            "input_mapping": {"x": "input.value"},
                            "dependencies": [],
                        },
                    }
                ]
            },
        }
    elif template == "example":
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "Example multi-step pipeline",
                "version": "1.0",
                "tags": ["example", "demo"],
            },
            "spec": {
                "ports": {
                    "llm": {
                        "adapter": "hexdag.builtin.adapters.mock.MockLLM",
                        "config": {},
                    }
                },
                "nodes": [
                    {
                        "kind": "hexdag.builtin.nodes.FunctionNode",
                        "metadata": {
                            "name": "input_processor",
                            "annotations": {"description": "Process input data"},
                        },
                        "spec": {
                            "fn": "my_module.process_input",
                            "input_mapping": {"data": "input.raw_data"},
                            "dependencies": [],
                        },
                    },
                    {
                        "kind": "hexdag.builtin.nodes.LLMNode",
                        "metadata": {
                            "name": "analyzer",
                            "annotations": {"description": "Analyze processed data"},
                        },
                        "spec": {
                            "prompt_template": "Analyze this data: {{input_processor}}",
                            "dependencies": ["input_processor"],
                        },
                    },
                    {
                        "kind": "hexdag.builtin.nodes.FunctionNode",
                        "metadata": {
                            "name": "output_formatter",
                            "annotations": {"description": "Format final output"},
                        },
                        "spec": {
                            "fn": "my_module.format_output",
                            "input_mapping": {"result": "analyzer"},
                            "dependencies": ["analyzer"],
                        },
                    },
                ],
            },
        }
    else:  # full
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "Full-featured pipeline template",
                "version": "1.0",
                "author": "Your Name",
                "tags": ["production", "template"],
            },
            "spec": {
                "input_schema": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
                "ports": {
                    "llm": {
                        "adapter": "hexdag.builtin.adapters.mock.MockLLM",
                        "config": {},
                    }
                },
                "common_field_mappings": {"default_mapping": {"value": "input.data"}},
                "nodes": [
                    {
                        "kind": "hexdag.builtin.nodes.FunctionNode",
                        "metadata": {
                            "name": "validator",
                            "annotations": {
                                "description": "Validate input",
                                "timeout": "5s",
                            },
                        },
                        "spec": {
                            "fn": "my_module.validate_input",
                            "input_mapping": {"data": "input.data"},
                            "output_schema": {
                                "type": "object",
                                "properties": {"valid": {"type": "boolean"}},
                            },
                            "dependencies": [],
                        },
                    },
                    {
                        "kind": "hexdag.builtin.nodes.LLMNode",
                        "metadata": {"name": "processor"},
                        "spec": {
                            "prompt_template": "Process: {{validator.result}}",
                            "output_schema": {
                                "type": "object",
                                "properties": {"processed": {"type": "string"}},
                            },
                            "dependencies": ["validator"],
                        },
                    },
                ],
            },
        }

    # Write to file
    with Path.open(output, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Created pipeline: {output}")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Edit {output} to customize your pipeline")
    console.print(f"  2. Validate: [bold]hexdag validate {output}[/bold]")
