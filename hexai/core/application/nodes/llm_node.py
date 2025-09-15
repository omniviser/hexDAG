"""LLM node factory for creating LLM-based pipeline nodes."""

from typing import Any

from pydantic import BaseModel

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from ..prompt import PromptInput
from .base_llm_node import BaseLLMNode


@node(name="llm_node", subtype=NodeSubtype.LLM, namespace="core")
class LLMNode(BaseLLMNode):
    """Simple factory for creating LLM-based nodes with rich template support.

    Inherits all common LLM functionality from BaseLLMNode. Provides clean interface for creating
    simple LLM nodes.
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
        """Create a NodeSpec with rich template features and auto-inferred input schema."""
        return cls()(
            name=name,
            template=template,
            output_schema=output_schema,
            deps=deps,
            **kwargs,
        )
