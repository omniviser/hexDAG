import json

import typer

app = typer.Typer(help="Validation & Lint commands")


@app.command("lint")
def lint(paths: list[str], json_output: bool = False) -> None:
    # dummy lint implementation
    result = {"status": "ok", "paths": paths}
    if json_output:
        print(json.dumps(result))
    else:
        print(f"Linted {len(paths)} paths: {paths}")


@app.command("schema")
def schema(json_output: bool = False) -> None:
    schema = {"namespace": "string", "action": "string", "payload": {}}
    if json_output:
        print(json.dumps(schema, indent=2))
    else:
        print(schema)


@app.command("validate")
def validate(file: str, json_output: bool = False) -> None:
    # dummy validation
    valid = True
    result = {"valid": valid}
    if json_output:
        print(json.dumps(result))
    else:
        print(f"Event payload valid: {valid}")
