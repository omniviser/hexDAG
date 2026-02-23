"""LLMNode - Unified LLM node for prompt building, API calls, and parsing.

This is the primary LLM node in hexdag, providing an n8n-style unified interface
for all LLM interactions. It combines:
- Prompt templating (Jinja2-style variable substitution)
- LLM API calls (via the llm port)
- Optional structured output parsing (JSON/Pydantic validation)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError

from hexdag.kernel.context import get_port
from hexdag.kernel.exceptions import ParseError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.prompt.template import PromptTemplate
from hexdag.kernel.ports.llm import Message, SupportsGeneration  # noqa: TC001
from hexdag.kernel.protocols import to_dict
from hexdag.kernel.utils.caching import KeyedCache
from hexdag.kernel.utils.node_timer import node_timer
from hexdag.kernel.validation.secure_json import SafeJSON

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from collections.abc import Callable

# Runtime imports needed for get_type_hints() to resolve annotations for SchemaGenerator
from hexdag.kernel.domain.dag import NodeSpec  # noqa: TC001
from hexdag.kernel.orchestration.prompt import PromptInput, TemplateType  # noqa: TC001

logger = get_logger(__name__)

# Convention: Parse strategy options for dropdown menus in Studio UI
ParseStrategy = Literal["json", "json_in_markdown", "yaml"]

# Cache for schema instruction strings keyed by Pydantic model class.
# Schema generation is deterministic per model class so this is safe.
_SCHEMA_INSTRUCTION_CACHE: KeyedCache[str] = KeyedCache()


def _convert_dicts_to_messages(message_dicts: list[dict[str, str]]) -> list[Message]:
    """Convert list of message dicts to Message objects."""
    return [Message(**msg) for msg in message_dicts]


class LLMNode(BaseNodeFactory):
    """Unified LLM node - prompt building, API calls, and optional parsing.

    This is the primary node for LLM interactions in hexdag. It provides a simple,
    n8n-style interface that handles the complete LLM workflow in a single node.

    Capabilities
    ------------
    1. **Prompt Templating**: Jinja2-style variable substitution ({{variable}})
    2. **LLM API Calls**: Calls the configured LLM port
    3. **Structured Output**: Optional JSON parsing with Pydantic validation
    4. **System Prompts**: Optional system message support
    5. **Message History**: Support for conversation context

    Port Capabilities
    -----------------
    Requires ``llm`` port implementing ``SupportsGeneration`` (validated at mount time).

    Examples
    --------
    Simple text generation::

        llm = LLMNode()
        spec = llm(
            name="summarizer",
            prompt_template="Summarize this text: {{text}}"
        )

    With system prompt::

        spec = llm(
            name="assistant",
            prompt_template="Answer: {{question}}",
            system_prompt="You are a helpful assistant."
        )

    Structured output with JSON parsing::

        from pydantic import BaseModel

        class Analysis(BaseModel):
            sentiment: str
            confidence: float
            keywords: list[str]

        spec = llm(
            name="analyzer",
            prompt_template="Analyze this text: {{text}}",
            output_schema=Analysis,
            parse_json=True
        )

    YAML Pipeline Usage::

        - kind: llm_node
          metadata:
            name: analyzer
          spec:
            prompt_template: "Analyze: {{input}}"
            system_prompt: "You are an analyst"
            parse_json: true
            output_schema:
              summary: str
              confidence: float
    """

    # Port capability table (validated at mount time by orchestrator)
    _hexdag_port_capabilities: ClassVar[dict[str, list[type]]] = {
        "llm": [SupportsGeneration],
    }

    # Studio UI metadata
    _hexdag_icon = "Brain"
    _hexdag_color = "#8b5cf6"  # violet-500

    # Explicit schema for Studio UI (excludes internal alias 'template')
    _yaml_schema = {
        "type": "object",
        "properties": {
            "prompt_template": {
                "type": "string",
                "description": "Template for the user prompt (Jinja2-style {{variable}} syntax)",
            },
            "output_schema": {
                "type": "object",
                "description": (
                    "Expected output schema for structured output validation. "
                    "Field values can be basic types (str, int, float, bool) "
                    "or sanitized types: currency, flexible_bool, score, "
                    "upper_str, lower_str, nullable_str, trimmed_str. "
                    "Custom types can be defined in spec.custom_types."
                ),
            },
            "system_prompt": {
                "type": "string",
                "description": "System message to prepend to the conversation",
            },
            "parse_json": {
                "type": "boolean",
                "default": False,
                "description": "Parse the LLM response as JSON",
            },
            "parse_strategy": {
                "type": "string",
                "enum": ["json", "json_in_markdown", "yaml"],
                "default": "json",
                "description": "JSON parsing strategy",
            },
        },
        "required": ["prompt_template"],
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LLMNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        prompt_template: PromptInput | str | None = None,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        system_prompt: str | None = None,
        parse_json: bool = False,
        parse_strategy: ParseStrategy = "json",
        deps: list[str] | None = None,
        template: PromptInput | str | None = None,  # Alias for prompt_template (YAML compat)
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a unified LLM node specification.

        Parameters
        ----------
        name : str
            Node name (must be unique in the graph)
        prompt_template : PromptInput | str
            Template for the user prompt. Supports Jinja2-style {{variable}} syntax.
            Can be a string or PromptTemplate/ChatPromptTemplate object.
            Can also be provided as 'template' (alias for YAML compatibility).
        output_schema : dict[str, Any] | type[BaseModel] | None, optional
            Expected output schema for structured output. If provided with parse_json=True,
            the LLM response will be parsed and validated against this schema.
        system_prompt : str | None, optional
            System message to prepend to the conversation.
        parse_json : bool, optional
            If True, parse the LLM response as JSON and validate against output_schema.
            Default is False (returns raw text).
        parse_strategy : str, optional
            JSON parsing strategy: "json", "json_in_markdown", or "yaml".
            Default is "json".
        deps : list[str] | None, optional
            List of dependency node names.
        **kwargs : Any
            Additional parameters passed to NodeSpec.

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution.

        Raises
        ------
        ValueError
            If parse_json is True but no output_schema is provided.

        Examples
        --------
        >>> llm = LLMNode()
        >>> spec = llm(
        ...     name="greeter",
        ...     prompt_template="Say hello to {{name}}",
        ...     system_prompt="You are friendly."
        ... )
        """
        # Handle template alias for YAML compatibility
        actual_template = prompt_template or template
        if actual_template is None:
            raise ValueError("prompt_template (or template) is required")

        if parse_json and output_schema is None:
            raise ValueError("output_schema is required when parse_json=True")

        # Prepare template
        prepared_template = self._prepare_template(actual_template)

        # Infer input schema from template variables
        input_schema = self.infer_input_schema_from_template(
            prepared_template, special_params={"context_history", "system_prompt"}
        )

        # Create output model if schema provided
        output_model: type[BaseModel] | None = None
        if output_schema is not None:
            output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Create the LLM wrapper function
        llm_wrapper = self._create_llm_wrapper(
            name=name,
            template=prepared_template,
            output_model=output_model,
            system_prompt=system_prompt,
            parse_json=parse_json,
            parse_strategy=parse_strategy,
        )

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=llm_wrapper,
            input_schema=input_schema,
            output_schema=output_schema if parse_json else None,
            deps=deps,
            **kwargs,
        )

    # Template Processing
    # -------------------

    @staticmethod
    def _prepare_template(template: PromptInput | str) -> TemplateType:
        """Convert string input to PromptTemplate if needed."""
        if isinstance(template, str):
            return PromptTemplate(template)
        return template

    # LLM Wrapper Creation
    # --------------------

    def _create_llm_wrapper(
        self,
        name: str,
        template: TemplateType,
        output_model: type[BaseModel] | None,
        system_prompt: str | None,
        parse_json: bool,
        parse_strategy: str,
    ) -> Callable[..., Any]:
        """Create an async LLM wrapper function."""

        async def llm_wrapper(validated_input: dict[str, Any]) -> Any:
            """Execute LLM call with optional parsing."""
            node_logger = logger.bind(node=name, node_type="llm_node")

            llm = get_port("llm")
            if not llm:
                raise RuntimeError("LLM port not available in execution context")

            with node_timer() as t:
                try:
                    # Convert input to dict if needed
                    try:
                        input_dict = to_dict(validated_input)
                    except TypeError:
                        input_dict = validated_input

                    # Log input variables at debug level
                    node_logger.debug(
                        "Prompt variables",
                        variables=list(input_dict.keys()),
                        variable_count=len(input_dict),
                    )

                    # Log execution start
                    node_logger.info(
                        "Calling LLM",
                        has_system_prompt=system_prompt is not None,
                        parse_json=parse_json,
                        parse_strategy=parse_strategy if parse_json else None,
                    )

                    # Enhance template with schema instructions if using structured output
                    enhanced_template = template
                    if parse_json and output_model:
                        enhanced_template = self._enhance_template_with_schema(
                            template, output_model
                        )

                    # Generate messages from template
                    messages = self._generate_messages(enhanced_template, input_dict, system_prompt)

                    # Call LLM
                    response = await llm.aresponse(messages)

                    node_logger.debug(
                        "LLM response received",
                        response_length=len(response) if isinstance(response, str) else None,
                        duration_ms=t.duration_str,
                    )

                    # Parse response if requested
                    if parse_json and output_model:
                        result = self._parse_response(response, output_model, parse_strategy)
                        node_logger.debug(
                            "Response parsed",
                            output_type=type(result).__name__,
                        )
                        return result

                    return response

                except Exception as e:
                    node_logger.error(
                        "LLM call failed",
                        duration_ms=t.duration_str,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

        return llm_wrapper

    def _generate_messages(
        self,
        template: TemplateType,
        input_data: dict[str, Any],
        system_prompt: str | None,
    ) -> list[Message]:
        """Generate messages from template and input data."""
        message_dicts = template.to_messages(**input_data)

        # Add system prompt if provided and not already present
        if system_prompt:
            has_system = any(msg.get("role") == "system" for msg in message_dicts)
            if not has_system:
                message_dicts.insert(0, {"role": "system", "content": system_prompt})

        return _convert_dicts_to_messages(message_dicts)

    # Schema Enhancement
    # ------------------

    def _enhance_template_with_schema(
        self, template: TemplateType, output_model: type[BaseModel]
    ) -> TemplateType:
        """Add schema instructions to template for structured output."""
        schema_instruction = self._create_schema_instruction(output_model)
        return template + schema_instruction

    def _create_schema_instruction(self, output_model: type[BaseModel]) -> str:
        """Create schema instruction for structured output.

        Cached per model class — schema generation is deterministic.
        """

        def _build() -> str:
            schema = output_model.model_json_schema()

            fields_info = []
            if "properties" in schema:
                for field_name, field_schema in schema["properties"].items():
                    field_type = field_schema.get("type", "any")
                    field_desc = field_schema.get("description", "")
                    desc_part = f" - {field_desc}" if field_desc else ""
                    fields_info.append(f"  - {field_name}: {field_type}{desc_part}")

            fields_text = (
                "\n".join(fields_info) if fields_info else "  - (no specific fields defined)"
            )

            example_data = {field: f"<{field}_value>" for field in schema.get("properties", {})}
            example_json = json.dumps(example_data, indent=2)

            return f"""

## Output Format
Respond with valid JSON matching this schema:
{fields_text}

Example: {example_json}
"""

        return _SCHEMA_INSTRUCTION_CACHE.get_or_create(output_model, _build)

    # Response Parsing
    # ----------------

    def _parse_response(
        self, response: str, output_model: type[BaseModel], strategy: str
    ) -> BaseModel:
        """Parse LLM response into structured output."""
        try:
            if strategy == "yaml":
                parsed_data = self._parse_yaml(response)
            else:
                # "json", "json_in_markdown", and default all use the same path
                parsed_data = self._parse_json(response)

        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            error_msg = self._create_parse_error_message(response, str(e), strategy)
            raise ParseError(error_msg) from e

        # Validate against schema
        try:
            return output_model.model_validate(parsed_data)
        except ValidationError as e:
            error_msg = self._create_validation_error_message(
                response, parsed_data, e, output_model
            )
            raise ParseError(error_msg) from e

    _safe_json = SafeJSON()

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text, including extraction from markdown code blocks."""
        result = self._safe_json.loads_from_text(text)
        if result.ok:
            return result.data  # type: ignore[return-value]
        # Fall back to direct parse for better error messages
        return json.loads(text.strip())  # type: ignore[no-any-return]

    def _parse_yaml(self, text: str) -> dict[str, Any]:
        """Parse YAML from text."""
        result = self._safe_json.loads_yaml_from_text(text)
        if result.ok:
            return result.data  # type: ignore[return-value]
        import yaml  # lazy: optional yaml dependency

        return yaml.safe_load(text)  # type: ignore[no-any-return]

    # Error Messages
    # --------------

    def _create_parse_error_message(self, text: str, error: str, strategy: str) -> str:
        """Create helpful error message for parse failures."""
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
"""

    def _create_validation_error_message(
        self,
        text: str,
        parsed_data: Any,
        error: ValidationError,
        model: type[BaseModel],
    ) -> str:
        """Create helpful error message for validation failures."""
        schema = model.model_json_schema()
        required_fields = schema.get("required", [])

        preview = str(parsed_data)[:200]

        return f"""
Parsed data does not match expected schema.

Expected schema: {model.__name__}
Required fields: {required_fields}

Parsed data preview:
{preview}

Validation errors:
{error}
"""

    # Legacy Compatibility
    # --------------------

    @classmethod
    def from_template(
        cls,
        name: str,
        template: PromptInput | str,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec from template (legacy compatibility method).

        This method provides backward compatibility with the old LLMNode API.
        """
        return cls()(
            name=name,
            prompt_template=template,
            output_schema=output_schema,
            parse_json=output_schema is not None,
            deps=deps,
            **kwargs,
        )
