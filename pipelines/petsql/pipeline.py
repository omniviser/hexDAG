"""PetSQL pipeline implementation following hexAI framework conventions."""

from ..base import PipelineDefinition
from ..shared_functions import extract_user_query, load_database_schema
from .functions import (
    convert_sqlite_to_postgres,
    filter_relevant_schema,
    select_examples,
    sql_validator,
)


class PetSQLPipeline(PipelineDefinition):
    """PetSQL text-to-SQL pipeline with 6-step process and PostgreSQL conversion."""

    @property
    def name(self) -> str:
        """Pipeline name."""
        return "petsql_pipeline"

    @property
    def description(self) -> str:
        """Pipeline description."""
        return "PetSQL text-to-SQL generation with 6-step process and PostgreSQL conversion"

    def _register_functions(self) -> None:
        """Register PetSQL pipeline functions."""
        # Step 1: Extract user query
        self.builder.register_function("extract_user_query", extract_user_query)

        # Step 2: Load database schema
        self.builder.register_function("load_database_schema", load_database_schema)

        # Step 3: Select examples using embedding similarity
        self.builder.register_function("select_examples", select_examples)

        # Step 4: Filter relevant schema
        self.builder.register_function("filter_relevant_schema", filter_relevant_schema)

        # Step 5: SQL validation and table extraction
        self.builder.register_function("sql_validator", sql_validator)

        # Step 8: Convert SQLite to PostgreSQL
        self.builder.register_function("convert_sqlite_to_postgres", convert_sqlite_to_postgres)
