import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import typer
from rich.console import Console


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
