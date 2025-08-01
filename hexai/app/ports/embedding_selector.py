"""Embedding selector port interface for text2sql example selection."""

from typing import Any, Protocol


class EmbeddingSelectorPort(Protocol):
    """Port interface for embedding-based example and schema selection.

    This port abstracts access to embedding-based similarity search for selecting relevant SQL
    examples and database schema elements.
    """

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
            - schema: Schema object or dictionary associated with the example
        """
        ...

    def get_relevant_tables(self, query_dict: dict[str, Any]) -> list[str]:
        """Get relevant table names based on query similarity.

        Args
        ----
            query_dict: Dictionary containing the question/query to match against

        Returns
        -------
            List of relevant table names
        """
        ...

    def is_available(self) -> bool:
        """Check if the embedding selector service is available.

        Returns
        -------
            True if the service is available and ready to use
        """
        ...
