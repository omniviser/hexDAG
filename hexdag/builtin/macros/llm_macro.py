"""LLM Macro - Structured LLM workflow with prompt and optional parsing.

This macro provides a convenient way to use the unified LLMNode with
structured output parsing in a declarative YAML-friendly format.

Note: With the unified LLMNode, this macro is now a thin wrapper.
Consider using LLMNode directly for simpler use cases.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from hexdag.builtin.nodes.llm_node import LLMNode
from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.orchestration.prompt import PromptInput
from hexdag.core.utils.schema_conversion import normalize_schema


class LLMMacroConfig(MacroConfig):
    """Configuration for LLM macro.

    Attributes
    ----------
    template : PromptInput
        Prompt template for LLM
    output_schema : dict[str, type] | type[BaseModel] | None
        Expected output schema (if None, returns raw text)
    system_prompt : str | None
        Optional system prompt
    parse_strategy : str
        Parsing strategy: "json", "json_in_markdown", "yaml"
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template: PromptInput
    output_schema: dict[str, type] | type[BaseModel] | None = None
    system_prompt: str | None = None
    parse_strategy: Literal["json", "json_in_markdown", "yaml"] = "json"

    @field_validator("output_schema", mode="before")
    @classmethod
    def normalize_output_schema(cls, v: Any) -> Any:
        """Convert YAML-friendly schema to Python types."""
        if v is None:
            return None
        # Use the shared utility to convert string type names to actual types
        return normalize_schema(v)


class LLMMacro(ConfigurableMacro):
    """LLM macro that wraps the unified LLMNode.

    This macro provides a YAML-friendly interface for structured LLM interactions.
    It uses the unified LLMNode which handles prompt templating, API calls, and
    optional JSON parsing in a single node.

    Note: For simple use cases, consider using LLMNode directly instead of this macro.

    Examples
    --------
    Basic usage (text generation)::

        from hexdag.builtin.macros import LLMMacro, LLMMacroConfig

        config = LLMMacroConfig(
            template="Explain {{topic}} in simple terms"
        )

        macro = LLMMacro(config)
        graph = macro.expand(
            instance_name="explainer",
            inputs={"topic": "quantum computing"},
            dependencies=[]
        )

    Structured output with parsing::

        from pydantic import BaseModel

        class Explanation(BaseModel):
            summary: str
            key_points: list[str]

        config = LLMMacroConfig(
            template="Explain {{topic}}. Return JSON with summary and key_points.",
            output_schema=Explanation,
            parse_strategy="json"
        )

        macro = LLMMacro(config)
        graph = macro.expand(...)

    YAML usage::

        nodes:
          - kind: macro_invocation
            metadata:
              name: analyzer
            spec:
              macro: core:llm_workflow
              config:
                template: "Analyze {{data}}"
                output_schema:
                  summary: str
                  sentiment: str
              inputs:
                data: "{{previous_node.output}}"
    """

    Config = LLMMacroConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> DirectedGraph:
        """Expand macro into a DirectedGraph with a single LLMNode.

        Args
        ----
            instance_name: Base name for generated nodes
            inputs: Input mappings for the macro
            dependencies: List of node names this macro depends on

        Returns
        -------
        DirectedGraph
            Graph containing a single unified LLMNode
        """
        config: LLMMacroConfig = self.config  # type: ignore[assignment]

        graph = DirectedGraph()

        # Create unified LLMNode
        llm_node_factory = LLMNode()
        llm_spec = llm_node_factory(
            name=instance_name,
            prompt_template=config.template,
            output_schema=config.output_schema,
            system_prompt=config.system_prompt,
            parse_json=config.output_schema is not None,
            parse_strategy=config.parse_strategy,
            deps=dependencies,
        )
        graph += llm_spec

        return graph
