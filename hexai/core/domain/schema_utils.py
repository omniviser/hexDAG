"""Schema utilities for template variable preparation and validation.

This module provides utilities for working with schemas and preparing template variables.
"""

from typing import Any

from pydantic import BaseModel


class SchemaUtils:
    """Utility class for schema-related operations."""

    @staticmethod
    def prepare_template_variables(input_data: Any) -> dict[str, Any]:
        """Prepare template variables from input data (Pydantic models only).

        Returns
        -------
            Dictionary of template variables.

        Raises
        ------
        TypeError
            If input data is not a dict or Pydantic BaseModel instance.
        """
        if isinstance(input_data, dict):
            return input_data.copy()
        if isinstance(input_data, BaseModel):
            return input_data.model_dump()
        raise TypeError("Input data must be a dict or Pydantic BaseModel instance.")
