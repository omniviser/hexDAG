import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable
from typing import Any

import click
import typer
import yaml
from rich.console import Console

from hexai.cli.utils import print_output


class NodeExecutionError(Exception):
    pass


app = typer.Typer()
console = Console()


class Pipeline:
    def __init__(self, nodes: dict[str, Any], edges: dict[str, list[str]]) -> None:
        """
        nodes: {node_name: Node}
        edges: {node_name: [downstream_node_names]}
        """
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def from_dict(cls, data: dict[str, Any], registry: Any) -> "Pipeline":
        # Build nodes from registry references
        nodes: dict[str, Any] = {}
        for n in data["components"]:
            node_name: str = n["name"]
            node_class: Any = registry.get(n["class"])
            nodes[node_name] = node_class(**n.get("config", {}))

        # Edges / dependencies
        edges: dict[str, list[str]] = defaultdict(list)
        for n in data["components"]:
            for dep in n.get("depends_on", []):
                edges[dep].append(n["name"])

        return cls(nodes, edges)

    async def execute(
        self, input_data: dict[str, Any], max_concurrency: int = 4
    ) -> AsyncGenerator[str, None]:
        # Simple topological execution
        pending: dict[str, Any] = dict(self.nodes)
        completed: set[str] = set()
        queue: list[str] = [
            n
            for n, node in pending.items()
            if not any(dep in pending for dep in getattr(node, "depends_on", []))
        ]

        sem = asyncio.Semaphore(max_concurrency)

        async def run_node(name: str, node: Any) -> AsyncGenerator[str, None]:
            async with sem:
                yield f"Running node: {name}"
                _ = (
                    await node.run(input_data)
                    if asyncio.iscoroutinefunction(node.run)
                    else node.run(input_data)
                )
                yield f"Node {name} completed"
                completed.add(name)
                # add new ready nodes
                queue.extend(
                    downstream
                    for downstream in self.edges.get(name, [])
                    if downstream in pending
                    and all(
                        dep in completed
                        for dep in getattr(self.nodes[downstream], "depends_on", [])
                    )
                )
                pending.pop(name)

        while queue:
            tasks = [run_node(n, pending[n]) for n in list(queue)]
            queue.clear()
            for coro in tasks:
                async for log in coro:
                    yield log


def _topological_layers(nodes: Iterable[dict[str, Any]]) -> list[list[str]]:
    # Build in-degree and adjacency
    deps = {n["name"]: set(n.get("depends_on", [])) for n in nodes}
    layers: list[list[str]] = []
    remaining = dict(deps)
    while remaining:
        ready = [n for n, ds in remaining.items() if not ds]
        if not ready:
            # cycle
            break
        layers.append(sorted(ready))
        for r in ready:
            remaining.pop(r)
        for k in remaining:
            remaining[k] = remaining[k] - set(ready)
    return layers


@app.command("validate")
def validate(pipeline_yaml: str) -> None:
    """Validate pipeline YAML for basic structure and DAG correctness."""
    with open(pipeline_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    errors: list[str] = []
    if "nodes" not in data or not isinstance(data["nodes"], list):
        errors.append("Missing or invalid 'nodes' list")
    else:
        names = set()
        for n in data["nodes"]:
            if "name" not in n or "func" not in n:
                errors.append(f"Node missing 'name' or 'func': {n}")
            else:
                if n["name"] in names:
                    errors.append(f"Duplicate node name: {n['name']}")
                names.add(n["name"])

    # Simple cycle check via DFS
    if "nodes" in data:
        graph: dict[str, list[str]] = {n["name"]: n.get("depends_on", []) for n in data["nodes"]}
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(u: str) -> bool:
            visited.add(u)
            stack.add(u)
            for v in graph.get(u, []):
                if v not in visited and dfs(v) or v in stack:
                    return True
            stack.remove(u)
            return False

        for node in graph:
            if node not in visited and dfs(node):
                errors.append("Cycle detected in node dependencies")
                break

    ctx = click.get_current_context()
    if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
        print_output({"valid": len(errors) == 0, "errors": errors}, ctx)
    else:
        if errors:
            for e in errors:
                console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
        console.print("[green]Pipeline validated OK[/green]")


@app.command("plan")
def plan(pipeline_yaml: str) -> None:
    """Show wave-based execution plan (topological layers)."""
    with open(pipeline_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    nodes = data.get("nodes", [])
    layers = _topological_layers(nodes)
    ctx = click.get_current_context()
    if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
        print_output({"layers": layers}, ctx)
    else:
        for i, layer in enumerate(layers):
            console.print(f"Wave {i}: {', '.join(layer)}")


@app.command("graph")
def graph(
    pipeline_yaml: str,
    out: str | None = typer.Option(None, "--out", help="Write DOT or image file"),
) -> None:
    """Render a simple DOT graph (outputs DOT text by default)."""
    with open(pipeline_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    nodes = data.get("nodes", [])
    edges: list[tuple[str, str]] = [
        (dep, n["name"]) for n in nodes for dep in n.get("depends_on", [])
    ]

    dot_lines = ["digraph pipeline {"]
    dot_lines.extend(f'  "{n["name"]}";' for n in nodes)
    dot_lines.extend(f'  "{a}" -> "{b}";' for a, b in edges)
    dot_lines.append("}")
    dot_text = "\n".join(dot_lines)

    if out:
        Path = __import__("pathlib").pathlib.Path
        Path(out).write_text(dot_text, encoding="utf-8")
        console.print(f"[green]Wrote DOT to {out}[/green]")
    else:
        ctx = click.get_current_context()
        if ctx.obj and ctx.obj.get("output_format") in ("json", "yaml"):
            print_output({"dot": dot_text}, ctx)
        else:
            console.print(dot_text)
