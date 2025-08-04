"""Adapter for text2sql selector implementations."""

from typing import Any

from text2sql.selectors import CosineSelector, CosineSimilarityQuestionMaskSelector, RandomSelector

from hexai.core.ports.embedding_selector import EmbeddingSelectorPort


class EmbeddingSelectorAdapter(EmbeddingSelectorPort):
    """Adapter that implements EmbeddingSelectorPort using existing selector classes."""

    def __init__(self, selector_instance: Any):
        """Initialize the adapter with a selector instance.

        Args
        ----
            selector_instance: Instance of a selector class (e.g., EmbeddingSelector)
        """
        self._selector = selector_instance

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
        try:
            return self._selector.get_examples(query_dict)
        except Exception:
            # Return empty list on error to gracefully handle failures
            return []

    def get_relevant_tables(self, query_dict: dict[str, Any]) -> list[str]:
        """Get relevant table names based on query similarity.

        Args
        ----
            query_dict: Dictionary containing the question/query to match against

        Returns
        -------
            List of relevant table names
        """
        try:
            examples = self._selector.get_examples(query_dict)
            # Extract table names from schema objects in examples
            table_names = []
            for _example_dict, schema in examples:
                if hasattr(schema, "get_table_names"):
                    table_names.extend(schema.get_table_names())
                elif isinstance(schema, dict):
                    table_names.extend(schema.keys())

            # Return unique table names
            return list(set(table_names))
        except Exception:
            return []

    def is_available(self) -> bool:
        """Check if the embedding selector service is available.

        Returns
        -------
            True if the service is available and ready to use
        """
        return self._selector is not None


def create_embedding_selector_adapter(
    selector_type: str = "CosineSimilarityQuestionMaskSelector",
    secrets: dict[str, Any] | None = None,
    k_shot: int = 5,
    **kwargs: Any,
) -> EmbeddingSelectorPort:
    """Create a text2sql selector adapter with configuration.

    Args
    ----
        selector_type: Type of selector to create (default: "CosineSimilarityQuestionMaskSelector")
        secrets: Configuration secrets for the selector
        k_shot: Number of examples to retrieve (default: 5)
        **kwargs: Additional selector-specific parameters

    Returns
    -------
        EmbeddingSelectorPort implementation
    """
    if secrets is None:
        secrets = {}

    try:
        # Create selector based on type
        if selector_type == "CosineSimilarityQuestionMaskSelector":
            selector = CosineSimilarityQuestionMaskSelector(
                k_shot=k_shot,
                masking_model=kwargs.get("masking_model", "gpt4o"),
                dataset=kwargs.get("dataset", "text2sql.data.Spider"),
                secrets=secrets,
            )
        elif selector_type == "CosineSelector":
            selector = CosineSelector(
                k_shot=k_shot,
                dataset=kwargs.get("dataset", "text2sql.data.Spider"),
                secrets=secrets,
            )
        elif selector_type == "RandomSelector":
            selector = RandomSelector(
                k_shot=k_shot, dataset=kwargs.get("dataset", "text2sql.data.Spider")
            )
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

        return EmbeddingSelectorAdapter(selector)
    except Exception:
        # Return a null adapter that gracefully handles failures
        return NullEmbeddingSelectorAdapter()


def create_selector_from_config(config: dict[str, Any]) -> EmbeddingSelectorPort:
    """Create a selector adapter from configuration dictionary.

    Args
    ----
        config: Configuration dictionary with selector settings
               Expected format from PetSQL config:
               {
                   "cls": "text2sql.selectors.CosineSimilarityQuestionMaskSelector",
                   "k_shot": 5,
                   "masking_model": "gpt4o",
                   "dataset": "text2sql.data.Spider"
               }

    Returns
    -------
        EmbeddingSelectorPort implementation
    """
    from app.utils.secret_utils.secret_loader import SecretsLoader

    try:
        # Extract selector class name
        cls_path = config.get("cls", "text2sql.selectors.CosineSimilarityQuestionMaskSelector")
        selector_type = cls_path.split(".")[-1]  # Get class name

        # Load secrets for the selector
        secrets_loader = SecretsLoader()
        try:
            secrets = secrets_loader.get_secret("SELECTORS", selector_type)
        except KeyError:
            secrets = {}

        # Create adapter with config
        return create_embedding_selector_adapter(
            selector_type=selector_type,
            secrets=secrets,
            k_shot=config.get("k_shot", 5),
            masking_model=config.get("masking_model", "gpt4o"),
            dataset=config.get("dataset", "text2sql.data.Spider"),
        )
    except Exception:
        return NullEmbeddingSelectorAdapter()


class NullEmbeddingSelectorAdapter:
    """Null object adapter for when embedding selector is unavailable."""

    def get_examples(self, query_dict: dict[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        """Get empty examples list when selector unavailable."""
        return []

    def get_relevant_tables(self, query_dict: dict[str, Any]) -> list[str]:
        """Get empty table list when selector unavailable."""
        return []

    def is_available(self) -> bool:
        """Check if null adapter is available (always False)."""
        return False
