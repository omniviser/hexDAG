"""Mock embedding selector port implementation for testing."""

from typing import Any

from hexai.core.ports.embedding_selector import EmbeddingSelectorPort


class MockEmbeddingSelectorPort(EmbeddingSelectorPort):
    """Mock implementation of EmbeddingSelectorPort for testing and demos."""

    def __init__(self) -> None:
        """Initialize with sample examples."""
        self._examples = [
            {
                "question": "Find all customers",
                "query": "SELECT * FROM customers;",
                "db_id": "sample_db",
            },
            {
                "question": "Count total orders",
                "query": "SELECT COUNT(*) FROM orders;",
                "db_id": "sample_db",
            },
            {
                "question": "Get customer orders with details",
                "query": (
                    "SELECT c.customer_name, o.order_date, o.order_value "
                    "FROM customers c JOIN orders o ON c.id = o.customer_id;"
                ),
                "db_id": "sample_db",
            },
            {
                "question": "Find customers with no orders",
                "query": (
                    "SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id "
                    "WHERE o.id IS NULL;"
                ),
                "db_id": "sample_db",
            },
            {
                "question": "Get top 10 orders by value",
                "query": "SELECT * FROM orders ORDER BY order_value DESC LIMIT 10;",
                "db_id": "sample_db",
            },
        ]

    def get_examples(self, query_dict: dict[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        """Get relevant examples based on query similarity.

        Args
        ----
            query_dict: Dictionary containing the question/query to match against
                       Expected format: {"question": "user's natural language query"}

        Returns
        -------
            List of tuples containing:
            - example_dict: Dictionary with example data (question, sql, etc.)
            - schema: Schema object or dictionary associated with the example (None for mock)
        """
        user_query = query_dict.get("question", "").lower()

        # Simple keyword-based matching for demo purposes
        relevant_examples = []

        for example in self._examples:
            example_question = example["question"].lower()
            # Check if any words from user query appear in example question
            user_words = set(user_query.split())
            example_words = set(example_question.split())

            # If there's overlap or if it's a general query, include the example
            if user_words.intersection(example_words) or len(user_words) <= 2:
                relevant_examples.append((example, None))  # None for schema

        # Return top 5 examples, or all if fewer than 5
        return relevant_examples[:5] if relevant_examples else [(self._examples[0], None)]

    def get_relevant_tables(self, query_dict: dict[str, Any]) -> list[str]:
        """Get relevant table names based on query similarity.

        Args
        ----
            query_dict: Dictionary containing the question/query to match against

        Returns
        -------
            List of relevant table names
        """
        user_query = query_dict.get("question", "").lower()

        # Simple keyword-based table detection
        tables = []
        if "customer" in user_query:
            tables.append("customers")
        if "order" in user_query:
            tables.append("orders")
        if "product" in user_query:
            tables.append("products")

        # Default fallback tables
        if not tables:
            tables = ["customers", "orders"]

        return tables

    def is_available(self) -> bool:
        """Check if the embedding selector service is available.

        Returns
        -------
            True if the service is available and ready to use
        """
        return True
