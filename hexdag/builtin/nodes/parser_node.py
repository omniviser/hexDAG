"""ParserNode - Structured output parsing with clear error messages.

This node handles ONLY output parsing - no LLM calls, no prompting.
Supports JSON parsing, Pydantic validation, and custom parsing strategies.
"""

import json
import re
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.exceptions import ParseError
from hexdag.core.logging import get_logger

from .base_node_factory import BaseNodeFactory

logger = get_logger(__name__)


class ParserInput(BaseModel):
    """Input model for ParserNode."""

    text: str
    """Raw text to parse (from LLM output)"""


class ParserNode(BaseNodeFactory):
    """Structured output parser with clear error messages and retry hints.

    This node:
    1. Accepts raw LLM output text
    2. Parses JSON or structured data
    3. Validates against Pydantic schemas
    4. Provides clear error messages for retry logic

    Architecture:
    ```
    PromptNode → RawLLMNode → ParserNode (validate & parse)
                                  ↓ (on error)
                            Retry with better prompt
    ```

    Examples
    --------
    Parse JSON to Pydantic model:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> parser = ParserNode()
        >>> spec = parser(name="person_parser", output_schema=Person)

    Parse with custom strategy:
        >>> spec = parser(
        ...     name="parser",
        ...     output_schema=Person,
        ...     strategy="json_in_markdown"  # Extract JSON from markdown blocks
        ... )

    Parse dict schema:
        >>> spec = parser(
        ...     name="parser",
        ...     output_schema={"name": str, "age": int}
        ... )
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ParserNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        output_schema: dict[str, type] | type[BaseModel],
        strategy: str = "json",
        strict: bool = True,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a parser node.

        Args
        ----
            name: Node name
            output_schema: Expected output schema (Pydantic model or dict)
            strategy: Parsing strategy - "json", "json_in_markdown", "yaml", "custom"
            strict: If True, raise errors on parse failure. If False, return partial results.
            deps: Dependencies
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            Configured node specification for parsing
        """
        # Convert dict schema to Pydantic model
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)
        if output_model is None:
            output_model = type(f"{name}Output", (BaseModel,), {"__annotations__": output_schema})

        # Create the parsing function
        parser_fn = self._create_parser(output_model, strategy=strategy, strict=strict)

        # Input is always ParserInput (text to parse)
        input_schema = {"text": str}

        # Use universal input mapping method
        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=parser_fn,
            input_schema=input_schema,
            output_schema=output_model,
            deps=deps,
            **kwargs,
        )

    def _create_parser(
        self, output_model: type[BaseModel], strategy: str = "json", strict: bool = True
    ) -> Any:
        """Create the parsing function.

        Args
        ----
            output_model: Target Pydantic model
            strategy: Parsing strategy
            strict: Strict validation mode

        Returns
        -------
        Callable
            Async function that parses text
        """

        async def parse_text(input_data: Any) -> BaseModel:
            """Parse text into structured output."""
            # Extract text from input
            if isinstance(input_data, dict):
                text = input_data.get("text", "")
            elif isinstance(input_data, BaseModel):
                text = getattr(input_data, "text", str(input_data))
            else:
                text = str(input_data)

            # Apply parsing strategy
            try:
                if strategy == "json":
                    parsed_data = self._parse_json(text)
                elif strategy == "json_in_markdown":
                    parsed_data = self._parse_json_in_markdown(text)
                elif strategy == "yaml":
                    parsed_data = self._parse_yaml(text)
                else:
                    # Fallback: try JSON first, then literal eval
                    parsed_data = self._parse_json(text)

            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                error_msg = self._create_parse_error_message(text, str(e), strategy)

                if strict:
                    raise ParseError(error_msg) from e

                # Non-strict mode: return empty model or partial data
                logger.warning(f"Parse failed (non-strict mode): {error_msg}")
                # Use model_construct to bypass validation
                return output_model.model_construct()

            # Validate against schema
            try:
                return output_model.model_validate(parsed_data)

            except ValidationError as e:
                error_msg = self._create_validation_error_message(
                    text, parsed_data, e, output_model
                )  # noqa: E501

                if strict:
                    raise ParseError(error_msg) from e

                # Non-strict mode: return best-effort model
                logger.warning(f"Validation failed (non-strict mode): {error_msg}")
                # Use model_construct to bypass validation
                return output_model.model_construct()

        return parse_text

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text.

        Args
        ----
            text: Raw text containing JSON

        Returns
        -------
        dict[str, Any]
            Parsed JSON data
        """
        # Strip whitespace and common markdown artifacts
        cleaned = text.strip()

        # Try direct JSON parsing
        try:
            parsed: dict[str, Any] = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text
            # Look for {...} or [...]
            json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if json_match:
                extracted: dict[str, Any] = json.loads(json_match.group(1))
                return extracted
            raise

    def _parse_json_in_markdown(self, text: str) -> dict[str, Any]:
        """Extract and parse JSON from markdown code blocks.

        Args
        ----
            text: Text containing markdown with JSON blocks

        Returns
        -------
        dict[str, Any]
            Parsed JSON data
        """
        # Extract from ```json ... ``` or ``` ... ```
        code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            # Try each code block
            for block in matches:
                try:
                    parsed: dict[str, Any] = json.loads(block)
                    return parsed
                except json.JSONDecodeError:
                    continue

        # Fallback to regular JSON parsing
        return self._parse_json(text)

    def _parse_yaml(self, text: str) -> dict[str, Any]:
        """Parse YAML from text.

        Args
        ----
            text: Raw text containing YAML

        Returns
        -------
        dict[str, Any]
            Parsed YAML data
        """
        parsed: dict[str, Any] = yaml.safe_load(text)
        return parsed

    def _create_parse_error_message(self, text: str, error: str, strategy: str) -> str:
        """Create helpful error message for parse failures.

        Args
        ----
            text: Original text that failed to parse
            error: Error message from parser
            strategy: Parsing strategy used

        Returns
        -------
        str
            Helpful error message with retry hints
        """
        preview = text[:200] + ("..." if len(text) > 200 else "")

        return f"""
Failed to parse LLM output using strategy '{strategy}'.

Error: {error}

Output preview:
{preview}

Retry hints:
1. Ensure the LLM output is valid {strategy.upper()} format
2. Check for trailing commas, missing quotes, or malformed syntax
3. Consider using 'json_in_markdown' strategy if JSON is in code blocks
4. Verify the output matches the expected schema

For retry logic: Update the prompt to request properly formatted {strategy.upper()} output.
"""

    def _create_validation_error_message(
        self, text: str, parsed_data: Any, error: ValidationError, model: type[BaseModel]
    ) -> str:
        """Create helpful error message for validation failures.

        Args
        ----
            text: Original text
            parsed_data: Data that failed validation
            error: Validation error
            model: Expected Pydantic model

        Returns
        -------
        str
            Helpful error message with schema hints
        """
        schema = model.model_json_schema()
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        preview = str(parsed_data)[:200]

        return f"""
Parsed data does not match expected schema.

Expected schema: {model.__name__}
Required fields: {required_fields}
Properties: {list(properties.keys())}

Parsed data preview:
{preview}

Validation errors:
{error}

Retry hints:
1. Update prompt to explicitly request these fields: {required_fields}
2. Provide example output showing the correct schema
3. Use few-shot examples with valid outputs
4. Ensure field names match exactly (case-sensitive)

Schema details:
{json.dumps(schema, indent=2)}
"""
