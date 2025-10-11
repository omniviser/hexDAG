"""LLM node factory for creating LLM-based pipeline nodes."""

from typing import Any

from pydantic import BaseModel

from hexdag.core.configurable import ConfigurableNode
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.prompt import PromptInput
from hexdag.core.registry import node
from hexdag.core.registry.models import NodeSubtype

from .base_llm_node import BaseLLMNode


@node(name="llm_node", subtype=NodeSubtype.LLM, namespace="core", required_ports=["llm"])
class LLMNode(BaseLLMNode, ConfigurableNode):
    """Simple factory for creating LLM-based nodes with rich template support.

    Inherits all common LLM functionality from BaseLLMNode. LLM nodes are highly dynamic -
    templates and schemas are passed via __call__() parameters rather than static Config.
    No Config class needed (follows YAGNI principle).
    """

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
