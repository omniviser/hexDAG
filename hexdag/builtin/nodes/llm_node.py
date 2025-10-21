"""LLM node factory for creating LLM-based pipeline nodes.

.. deprecated:: 0.2.0
   Use the new composable architecture instead:
   - PromptNode for prompt building
   - RawLLMNode for API calls
   - ParserNode for output parsing
   - Or use the LLM macro that combines all three with retry logic
"""

import warnings
from typing import Any

from pydantic import BaseModel

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.prompt import PromptInput
from hexdag.core.registry import node
from hexdag.core.registry.models import NodeSubtype

from .base_llm_node import BaseLLMNode


@node(name="llm_node", subtype=NodeSubtype.LLM, namespace="core", required_ports=["llm"])
class LLMNode(BaseLLMNode):
    """DEPRECATED: Use PromptNode + RawLLMNode + ParserNode instead.

    This monolithic LLM node combines prompt building, API calls, and parsing.
    The new architecture separates these concerns for better composability:

    Old (monolithic)::

        llm_node = LLMNode()
        spec = llm_node(name="analyzer", template="Analyze {{data}}", output_schema=Result)

    New (composable)::

        # Option 1: Use the LLM macro (recommended)
        from hexdag.builtin.macros import LLMMacro
        macro = LLMMacro()
        graph = macro.expand(
            instance_name="analyzer",
            inputs={"template": "Analyze {{data}}", "output_schema": Result}
        )

        # Option 2: Compose manually for full control
        prompt_node = PromptNode()(name="prompt", template="Analyze {{data}}")
        llm_node = RawLLMNode()(name="llm")
        parser_node = ParserNode()(name="parser", output_schema=Result)

    Inherits all common LLM functionality from BaseLLMNode. LLM nodes are highly dynamic -
    templates and schemas are passed via __call__() parameters rather than static Config.
    No Config class needed (follows YAGNI principle).

    .. deprecated:: 0.2.0
       Will be removed in version 0.3.0. Use the new composable architecture.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LLMNode with deprecation warning."""
        warnings.warn(
            "LLMNode is deprecated and will be removed in version 0.3.0. "
            "Use the new composable architecture: PromptNode + RawLLMNode + ParserNode, "
            "or use the LLM macro for automatic composition with retry logic. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    def __call__(
        self,
        name: str,
        template: PromptInput,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for an LLM-based node with rich template support.

        Args
        ----
            name: Node name
            template: Template for LLM prompt
            output_schema: Output schema for validation
            deps: List of dependency node names
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution

        Raises
        ------
        ValueError
            If output_schema is provided for string templates
        """
        # String templates don't support rich features (structured output)
        if isinstance(template, str) and output_schema is not None:
            raise ValueError(
                "output_schema not supported for string templates - "
                "use PromptTemplate, ChatPromptTemplate, or other template objects "
                "for structured output"
            )

        # Rich features enabled for non-string templates with output schema
        rich_features = not isinstance(template, str) and output_schema is not None

        # Use BaseLLMNode pipeline
        return self.build_llm_node_spec(
            name=name,
            template=template,
            output_schema=output_schema,
            deps=deps,
            rich_features=rich_features,
            **kwargs,
        )

    @classmethod
    def from_template(
        cls,
        name: str,
        template: PromptInput,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec with rich template features and auto-inferred input schema.

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution
        """
        return cls()(
            name=name,
            template=template,
            output_schema=output_schema,
            deps=deps,
            **kwargs,
        )
