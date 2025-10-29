"""BaseLLMNode - Foundation class for all LLM-based nodes."""

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from hexdag.core.context import get_port
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.prompt import PromptInput, TemplateType
from hexdag.core.orchestration.prompt.template import PromptTemplate
from hexdag.core.protocols import to_dict
from hexdag.core.validation.unified_engine import UnifiedParsingEngine

from .base_node_factory import BaseNodeFactory


class BaseLLMNode(BaseNodeFactory):
    """Base class for all LLM-based nodes with common functionality.

    Provides:
    - Template preparation and enhancement
    - Input schema inference from templates
    - LLM wrapper creation with event emission
    - Message generation from templates
    - Schema instructions for structured output
    - Standard NodeSpec building pipeline
    """

    # Template Processing Methods
    @staticmethod
    def prepare_template(template: PromptInput) -> TemplateType:
        """Convert string input to PromptTemplate if needed.

        Returns
        -------
        TemplateType
            The prepared template (either the original or a new PromptTemplate)
        """
        if isinstance(template, str):
            return PromptTemplate(template)
        return template

    @staticmethod
    def infer_input_schema_from_template(
        template: str | TemplateType, special_params: set[str] | None = None
    ) -> dict[str, Any]:
        """Infer input schema from template variables.

        Parameters
        ----------
        template : str | TemplateType
            Template to infer schema from
        special_params : set[str] | None
            Set of special parameter names to exclude from schema

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their types
        """
        # Use the shared implementation from BaseNodeFactory
        if special_params is None:
            special_params = {"context_history", "system_prompt"}
        return BaseNodeFactory.infer_input_schema_from_template(
            template, special_params=special_params
        )

    def enhance_template_with_schema(
        self, template: TemplateType, output_model: type[BaseModel]
    ) -> TemplateType:
        """Add schema instructions to template for structured output.

        Returns
        -------
        TemplateType
            The enhanced template with schema instructions
        """
        schema_instruction = self._create_schema_instruction(output_model)
        return template + schema_instruction

    def create_llm_wrapper(
        self,
        name: str,
        template: TemplateType,
        input_model: type[BaseModel] | None,
        output_model: type[BaseModel] | None,
        rich_features: bool = True,
    ) -> Callable[..., Any]:
        """Create an LLM wrapper function with event emission.

        Returns
        -------
        Callable[..., Any]
            Async wrapper function for LLM execution
        """

        async def llm_wrapper(validated_input: dict[str, Any]) -> Any:
            """Execute LLM call with proper event emission."""
            llm = get_port("llm")  # Validated by orchestrator

            try:
                enhanced_template = template
                if rich_features and output_model:
                    enhanced_template = self.enhance_template_with_schema(template, output_model)

                # Note: signature is dict[str, Any] but runtime may pass Pydantic models
                try:
                    input_dict = to_dict(validated_input)
                except TypeError:
                    # Already compatible type, use as-is
                    input_dict = validated_input

                messages, template_vars = self._generate_messages(enhanced_template, input_dict)

                # Call LLM
                if rich_features and output_model:
                    response = await llm.aresponse(messages)
                    result = self._parse_structured_response(response, output_model)
                else:
                    response = await llm.aresponse(messages)
                    result = response

                return result

            except Exception:
                # Re-raise to preserve stack trace and let orchestrator handle
                raise

        return llm_wrapper

    def _generate_messages(
        self, template: TemplateType, validated_input: dict[str, Any]
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Generate messages from template and extract variables.

        Returns
        -------
        tuple[list[dict[str, str]], dict[str, Any]]
            Tuple containing (messages, template_variables)
        """
        messages = template.to_messages(**validated_input)
        return messages, validated_input

    # Remove the individual message generation methods - they're not needed
    # All templates support to_messages() method

    def _create_schema_instruction(self, output_model: type[BaseModel]) -> str:
        """Create schema instruction for structured output.

        Returns
        -------
        str
            Formatted schema instruction text
        """
        schema = output_model.model_json_schema()

        fields_info = []
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                field_type = field_schema.get("type", "any")
                field_desc = field_schema.get("description", "")
                desc_part = f" - {field_desc}" if field_desc else ""
                fields_info.append(f"  - {field_name}: {field_type}{desc_part}")

        fields_text = "\n".join(fields_info) if fields_info else "  - (no specific fields defined)"

        example_data = {field: f"<{field}_value>" for field in schema.get("properties", {})}
        example_json = json.dumps(example_data, indent=2)

        return f"""

## Output Format
Respond with valid JSON matching this schema:
{fields_text}

Example: {example_json}
"""

    # Standard Node Building Pipeline
    def build_llm_node_spec(
        self,
        name: str,
        template: PromptInput,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        rich_features: bool = True,
        **kwargs: Any,
    ) -> NodeSpec:
        """Build a standard LLM NodeSpec with consistent patterns.

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution
        """
        # Prepare template
        prepared_template = self.prepare_template(template)

        # Infer input schema
        input_schema = self.infer_input_schema_from_template(prepared_template)

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = (
            self.create_pydantic_model(f"{name}Output", output_schema)
            if rich_features and output_schema
            else None
        )

        llm_wrapper = self.create_llm_wrapper(
            name, prepared_template, input_model, output_model, rich_features
        )

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=llm_wrapper,
            input_schema=input_schema,
            output_schema=output_schema if rich_features else None,
            deps=deps,
            **kwargs,
        )

    def _parse_structured_response(self, response: str, output_model: type[BaseModel]) -> Any:
        engine = UnifiedParsingEngine()
        res = engine.auto_detect_and_parse(response, output_model)
        if res.ok and res.data is not None:
            return res.data
        try:
            return output_model.model_validate({"result": response})
        except Exception:
            return response
