"""LLM Macro - Composable LLM workflow with prompt, LLM call, and parsing.

This macro combines PromptNode + RawLLMNode + ParserNode into a single
workflow for structured LLM interactions.

Architecture:
    PromptNode → RawLLMNode → ParserNode (optional)
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from hexdag.builtin.nodes.parser_node import ParserNode
from hexdag.builtin.nodes.prompt_node import PromptNode
from hexdag.builtin.nodes.raw_llm_node import RawLLMNode
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
    output_format : str
        Prompt output format: "messages" or "string"
    system_prompt : str | None
        Optional system prompt
    parse_strategy : str
        Parsing strategy: "json", "json_in_markdown", "yaml"
    strict_parsing : bool
        If True, raise errors on parse failure
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template: PromptInput
    output_schema: dict[str, type] | type[BaseModel] | None = None
    output_format: str = "messages"
    system_prompt: str | None = None
    parse_strategy: str = "json"
    strict_parsing: bool = False

    @field_validator("output_schema", mode="before")
    @classmethod
    def normalize_output_schema(cls, v: Any) -> Any:
        """Convert YAML-friendly schema to Python types."""
        if v is None:
            return None
        # Use the shared utility to convert string type names to actual types
        return normalize_schema(v)


class LLMMacro(ConfigurableMacro):
    """LLM macro that composes PromptNode + RawLLMNode + ParserNode.

    This macro replaces the monolithic LLMNode with a composable architecture:
    - PromptNode builds the prompt from templates
    - RawLLMNode calls the LLM API
    - ParserNode parses the output (optional)

    Benefits:
    1. **Separation of Concerns** - Each node has one responsibility
    2. **Composable** - Mix and match components
    3. **Clear Errors** - Parser provides helpful error messages
    4. **Type Safety** - Automatic schema validation

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
        """Expand macro into a DirectedGraph of nodes.

        Args
        ----
            instance_name: Base name for generated nodes
            inputs: Input mappings for the macro
            dependencies: List of node names this macro depends on

        Returns
        -------
        DirectedGraph
            Graph containing PromptNode → RawLLMNode → (ParserNode)
        """
        config: LLMMacroConfig = self.config  # type: ignore[assignment]

        graph = DirectedGraph()

        # Node 1: PromptNode - builds the prompt
        prompt_node_factory = PromptNode()
        prompt_spec = prompt_node_factory(
            name=f"{instance_name}_prompt",
            template=config.template,
            output_format=config.output_format,
            system_prompt=config.system_prompt,
            deps=dependencies,
        )
        graph += prompt_spec

        # Node 2: RawLLMNode - calls LLM API
        llm_node_factory = RawLLMNode()
        llm_spec = llm_node_factory(
            name=f"{instance_name}_llm",
            deps=[f"{instance_name}_prompt"],
        )
        graph += llm_spec

        # Node 3: ParserNode (optional) - parses structured output
        if config.output_schema is not None:
            parser_node_factory = ParserNode()
            parser_spec = parser_node_factory(
                name=f"{instance_name}_parser",
                output_schema=config.output_schema,
                strategy=config.parse_strategy,
                strict=config.strict_parsing,
                deps=[f"{instance_name}_llm"],
            )
            graph += parser_spec

        return graph
