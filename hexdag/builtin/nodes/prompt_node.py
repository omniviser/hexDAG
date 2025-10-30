"""PromptNode - Composable prompt building with registry support.

This node handles ONLY prompt construction - no LLM calls, no parsing.
Supports composition: ChatPrompt + FewShot + ToolPrompt + custom templates.
"""

from typing import Any

from pydantic import BaseModel

from hexdag.builtin.prompts.base import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.prompt import PromptInput
from hexdag.core.registry import node
from hexdag.core.registry.models import NodeSubtype

from .base_node_factory import BaseNodeFactory


@node(name="prompt_node", subtype=NodeSubtype.FUNCTION, namespace="core")
class PromptNode(BaseNodeFactory):
    """Composable prompt building node - LangChain-style prompt construction.

    This node:
    1. Accepts template inputs (string, PromptTemplate, ChatPromptTemplate, etc.)
    2. Supports composition via builder pattern
    3. Renders prompts with variable substitution
    4. Returns formatted messages or strings for LLM consumption

    Architecture:
    ```
    PromptNode (build) → RawLLMNode (call API) → ParserNode (parse output)
    ```

    Examples
    --------
    Simple string template:
        >>> prompt_node = PromptNode()
        >>> spec = prompt_node(name="greeter", template="Hello {{name}}!")

    Chat template:
        >>> from hexdag.builtin.prompts.base import ChatPromptTemplate
        >>> template = ChatPromptTemplate(
        ...     system_message="You are a helpful assistant",
        ...     human_message="Answer: {{question}}"
        ... )
        >>> spec = prompt_node(name="qa", template=template)

    Composed template (builder pattern):
        >>> base = PromptTemplate("Analyze {{data}}")
        >>> tools = PromptTemplate("\\n\\nAvailable tools: {{tools}}")
        >>> spec = prompt_node(name="analyzer", template=base + tools)

    Registry-based template reference:
        >>> # Reference a registered prompt by name
        >>> spec = prompt_node(name="agent", template="core:tool_usage_function")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize PromptNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        template: PromptInput | str | None = None,
        prompt_ref: str | None = None,
        prompt_args: dict[str, Any] | None = None,
        output_format: str = "messages",
        system_prompt: str | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a prompt building node.

        Args
        ----
            name: Node name
            template: Custom inline prompt template (string or PromptTemplate)
            prompt_ref: Registry reference to registered prompt (e.g., "core:chat_qa")
            prompt_args: Arguments to pass to registered prompt constructor
            output_format: "messages" (list[dict]) or "string" (str)
            system_prompt: Optional system prompt to prepend
            deps: Dependencies
            **kwargs: Additional parameters (passed to prompt if prompt_ref used)

        Returns
        -------
        NodeSpec
            Configured node specification for prompt building

        Examples
        --------
        Custom inline template::

            spec = prompt_node(
                name="greeter",
                template="Hello {{name}}!"
            )

        Registry reference::

            spec = prompt_node(
                name="qa",
                prompt_ref="core:chat_qa",
                prompt_args={}  # Constructor args for ChatQAPrompt
            )

        Registry with inline args::

            spec = prompt_node(
                name="classifier",
                prompt_ref="core:fewshot_classification",
                examples=[
                    {"input": "Great!", "output": "positive"},
                    {"input": "Bad", "output": "negative"}
                ]
            )
        """
        # Build the template: either from registry or custom
        if prompt_ref:
            # Load from registry and instantiate
            resolved_template = self._build_from_registry(prompt_ref, prompt_args or {}, kwargs)
        elif template:
            # Use custom template directly
            resolved_template = template
        else:
            raise ValueError("Must provide either 'template' or 'prompt_ref'")

        # Infer input schema from template
        input_schema = self._infer_input_schema(resolved_template)

        # Create the prompt building function
        prompt_fn = self._create_prompt_builder(
            resolved_template, output_format=output_format, system_prompt=system_prompt
        )

        # Define output schema based on format
        if output_format == "messages":
            output_schema = type(
                f"{name}Output",
                (BaseModel,),
                {"__annotations__": {"messages": list[dict[str, str]]}},
            )
        else:
            output_schema = type(f"{name}Output", (BaseModel,), {"__annotations__": {"text": str}})

        # Use universal input mapping method
        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=prompt_fn,
            input_schema=input_schema,
            output_schema=output_schema,
            deps=deps,
            **kwargs,
        )

    def _build_from_registry(
        self, prompt_ref: str, prompt_args: dict[str, Any], extra_kwargs: dict[str, Any]
    ) -> PromptInput:
        """Build prompt from registry reference.

        Args
        ----
            prompt_ref: Registry reference (e.g., "core:chat_qa")
            prompt_args: Arguments for prompt constructor
            extra_kwargs: Additional kwargs from __call__

        Returns
        -------
        PromptInput
            Instantiated prompt template

        Raises
        ------
        ValueError
            If prompt not found in registry
        """
        from hexdag.core.registry import registry

        # Parse registry reference
        if ":" in prompt_ref:
            namespace, name = prompt_ref.split(":", 1)
        else:
            namespace = "core"
            name = prompt_ref

        # Get prompt class from registry
        try:
            from hexdag.core.registry.models import ComponentType

            prompt_metadata = registry.get_metadata(
                name, namespace=namespace, component_type=ComponentType.PROMPT
            )
            prompt_class = prompt_metadata.component
        except Exception as e:
            raise ValueError(
                f"Prompt '{prompt_ref}' not found in registry. "
                f"Available: {list(registry._components.keys())}"
            ) from e

        # Merge prompt_args with extra_kwargs
        all_args = {**prompt_args, **extra_kwargs}

        # Instantiate the prompt
        return prompt_class(**all_args)  # type: ignore[operator,no-any-return]

    def _infer_input_schema(self, template: PromptInput) -> dict[str, Any]:
        """Infer input schema from template variables.

        Args
        ----
            template: Prompt template

        Returns
        -------
        dict[str, Any]
            Inferred input schema
        """
        return BaseNodeFactory.infer_input_schema_from_template(template, special_params=None)

    def _create_prompt_builder(
        self,
        template: PromptInput,
        output_format: str = "messages",
        system_prompt: str | None = None,
    ) -> Any:
        """Create the prompt building function.

        Args
        ----
            template: Prompt template
            output_format: "messages" or "string"
            system_prompt: Optional system prompt

        Returns
        -------
        Callable
            Async function that builds prompts
        """

        async def build_prompt(input_data: Any) -> dict[str, Any]:
            """Build prompt from template and input data."""
            # Convert input to dict for template rendering
            if isinstance(input_data, BaseModel):
                kwargs = input_data.model_dump()
            elif isinstance(input_data, dict):
                kwargs = input_data
            else:
                kwargs = {"input": str(input_data)}

            # Handle different template types
            if isinstance(template, str):
                # Simple string template
                prompt_template = PromptTemplate(template)
                rendered = prompt_template.render(**kwargs)

                if output_format == "messages":
                    messages = [{"role": "user", "content": rendered}]
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    return {"messages": messages}
                return {"text": rendered}

            if isinstance(template, (ChatPromptTemplate, ChatFewShotTemplate)):
                # Chat-style template with role-based messages
                # Filter kwargs to only pass valid arguments to to_messages
                to_messages_kwargs: dict[str, Any] = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["context_history"] or isinstance(v, list)
                }
                if "context_history" in kwargs and isinstance(kwargs["context_history"], list):
                    # Type narrowing: we checked it's a list
                    to_messages_kwargs["context_history"] = kwargs["context_history"]
                messages = template.to_messages(system_prompt=system_prompt, **to_messages_kwargs)
                if output_format == "messages":
                    return {"messages": messages}
                # Convert messages to string format
                text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
                return {"text": text}

            if isinstance(template, (PromptTemplate, FewShotPromptTemplate)):
                # Standard prompt template
                rendered = template.format(**kwargs)

                if output_format == "messages":
                    messages = [{"role": "user", "content": rendered}]
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    return {"messages": messages}
                return {"text": rendered}

            # Should never reach here given PromptInput type constraint
            raise TypeError(f"Unsupported template type: {type(template)}")

        return build_prompt
