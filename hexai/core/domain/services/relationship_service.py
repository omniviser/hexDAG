"""Domain service for managing table relationships and graph operations.

This service contains pure domain logic for working with database table relationships, including
graph construction and pathfinding algorithms.
"""

from collections import defaultdict, deque
from typing import Any


class RelationshipService:
    """Domain service for table relationship graph operations."""

    def __init__(self) -> None:
        """Initialize the relationship service."""
        self._relationship_graph: dict[str, set[str]] | None = None

    def build_relationship_graph(self, relationships: list[dict[str, Any]]) -> dict[str, set[str]]:
        """Build bidirectional relationship graph from foreign keys.

        Args
        ----
        relationships: List of relationship data from database

        Returns
        -------
        Graph where keys are table names and values are sets of connected tables
        """
        graph: dict[str, set[str]] = defaultdict(set)

        for rel in relationships:
            if isinstance(rel, dict):
                from_table = rel.get("from", "").split(".")[0]
                to_table = rel.get("to", "").split(".")[0]
            else:
                # Handle string format like "orders.customer_id -> customers.id"
                rel_str = str(rel)
                if " -> " in rel_str:
                    from_part, to_part = rel_str.split(" -> ")
                    from_table = from_part.split(".")[0]
                    to_table = to_part.split(".")[0]
                else:
                    continue

            if from_table and to_table:
                graph[from_table].add(to_table)
                graph[to_table].add(from_table)

        return dict(graph)

    def get_tables_with_relationships(
        self, core_tables: list[str], relationships: list[dict[str, Any]]
    ) -> list[str]:
        """Get core tables plus all intermediate tables that connect them.

        Args
        ----
        core_tables: List of core table names
        relationships: List of relationship data from database

        Returns
        -------
        List of all tables including intermediates
        """
        if not core_tables:
            return core_tables

        # Build relationship graph
        graph = self.build_relationship_graph(relationships)

        # Find all intermediate tables
        intermediate_tables = set()

        for i, table1 in enumerate(core_tables):
            for table2 in core_tables[i + 1 :]:  # noqa: E203
                path_tables = self.find_path_tables(table1, table2, graph)
                intermediate_tables.update(path_tables)

        # Combine core and intermediate tables
        return list(set(core_tables) | intermediate_tables)

    def find_path_tables(
        self, start_table: str, end_table: str, graph: dict[str, set[str]]
    ) -> set[str]:
        """Find all tables in the shortest path between two tables.

        Args
        ----
        start_table: Starting table name
        end_table: Ending table name
        graph: Relationship graph

        Returns
        -------
        Set of table names in the path (including start and end)
        """
        if start_table == end_table:
            return {start_table}

        if start_table not in graph or end_table not in graph:
            return set()

        # BFS to find shortest path
        queue = deque([(start_table, [start_table])])
        visited = {start_table}

        while queue:
            current_table, path = queue.popleft()

            if current_table == end_table:
                return set(path)

            for neighbor in graph.get(current_table, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return set()  # No path found

    def get_connected_tables(self, table_names: list[str], graph: dict[str, set[str]]) -> set[str]:
        """Get all tables that are directly connected to the given tables.

        Args
        ----
        table_names: List of table names to find connections for
        graph: Relationship graph

        Returns
        -------
        Set of all connected table names (including the input tables)
        """
        connected = set(table_names)

        for table in table_names:
            if table in graph:
                connected.update(graph[table])

        return connected

    def is_connected(self, table1: str, table2: str, graph: dict[str, set[str]]) -> bool:
        """Check if two tables are connected through relationships.

        Args
        ----
        table1: First table name
        table2: Second table name
        graph: Relationship graph

        Returns
        -------
        True if tables are connected, False otherwise
        """
        return len(self.find_path_tables(table1, table2, graph)) > 0

    def get_relationship_distance(
        self, table1: str, table2: str, graph: dict[str, set[str]]
    ) -> int:
        """Get the relationship distance (number of hops) between two tables.

        Args
        ----
        table1: First table name
        table2: Second table name
        graph: Relationship graph

        Returns
        -------
        Number of relationship hops, -1 if not connected
        """
        path = self.find_path_tables(table1, table2, graph)
        return len(path) - 1 if path else -1

    def get_relationships(
        self, core_tables: list[str], relationships: list[dict[str, Any]]
    ) -> list[str]:
        """Get core tables plus all intermediate tables that connect them.

        Args
        ----
        core_tables: List of core table names
        relationships: List of relationship data from database

        Returns
        -------
        List of all tables including intermediates
        """
        if not core_tables:
            return core_tables

        # Build relationship graph
        graph = self.build_relationship_graph(relationships)

        # Find all intermediate tables
        intermediate_tables = set()

        for i, table1 in enumerate(core_tables):
            for table2 in core_tables[i + 1 :]:  # noqa: E203
                path_tables = self.find_path_tables(table1, table2, graph)
                intermediate_tables.update(path_tables)

        # Combine core and intermediate tables
        return list(set(core_tables) | intermediate_tables)
