"""LLMNode - Unified LLM node for prompt building, API calls, and parsing.

This is the primary LLM node in hexdag, providing a unified interface
for all LLM interactions. It combines:
- Prompt templating (``{{variable}}`` substitution)
- LLM API calls (via the llm port)
- Optional structured output (native or fallback via port middleware)
- Conversation history assembly from the DAG
- Few-shot examples
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

from hexdag.kernel.context import get_pipeline_name, get_port
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.prompt.template import PromptTemplate
from hexdag.kernel.ports.llm import Message, SupportsGeneration  # noqa: TC001
from hexdag.kernel.protocols import to_dict
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

# Runtime imports needed for get_type_hints() to resolve annotations for SchemaGenerator
from hexdag.kernel.domain.dag import NodeSpec  # noqa: TC001
from hexdag.kernel.orchestration.prompt import PromptInput, TemplateType  # noqa: TC001

logger = get_logger(__name__)


def _convert_dicts_to_messages(message_dicts: list[dict[str, str]]) -> list[Message]:
    """Convert list of message dicts to Message objects."""
    return [Message.model_validate(msg) for msg in message_dicts]


def _format_examples(examples: list[dict[str, Any]]) -> str:
    """Format few-shot examples as text for the system message."""
    lines: list[str] = ["\n\nExamples:"]
    for ex in examples:
        inp = ex.get("input", "")
        out = ex.get("output", "")
        lines.append(f"Input: {inp}\nOutput: {out}")
    return "\n\n".join(lines)


def _resolve_conversation(
    conversation: str | list[Any],
    inputs: dict[str, Any],
) -> list[Message]:
    """Resolve conversation field into a list of Message objects.

    Supports three forms:
    1. String ``"{{node}}"`` — resolve from inputs (expects list[dict])
    2. List of ``{prompt, messages}`` — framing prompts + dependency refs
    3. List of ``{role, content}`` — inline static messages
    """
    import re  # lazy: only needed for conversation resolution

    if isinstance(conversation, str):
        # Single dependency reference: "{{node_name}}"
        match = re.match(r"\{\{\s*(\w+)\s*\}\}", conversation)
        if match:
            var_name = match.group(1)
            raw = inputs.get(var_name)
            if raw is None:
                raise ValueError(
                    f"Conversation reference '{{{{{var_name}}}}}' not found in inputs. "
                    f"Available: {list(inputs.keys())}"
                )
            if not isinstance(raw, list):
                raise TypeError(
                    f"Conversation reference '{{{{{var_name}}}}}' must resolve to a list, "
                    f"got {type(raw).__name__}"
                )
            return [Message.model_validate(m) for m in raw]
        # Treat as literal text (edge case)
        return [Message(role="user", content=conversation)]

    if isinstance(conversation, list) and conversation:
        first = conversation[0]
        if isinstance(first, dict):
            # Detect form: inline static [{role, content}]
            if "role" in first:
                return [Message.model_validate(m) for m in conversation]

            # Detect form: multi-source [{prompt, messages}]
            if "messages" in first or "prompt" in first:
                result: list[Message] = []
                for block in conversation:
                    if not isinstance(block, dict):
                        continue
                    # Add framing prompt
                    if prompt := block.get("prompt"):
                        result.append(Message(role="user", content=prompt))
                    # Resolve messages reference
                    if msg_ref := block.get("messages"):
                        resolved = _resolve_conversation(msg_ref, inputs)
                        result.extend(resolved)
                return result

    return []


class LLMNode(BaseNodeFactory, yaml_alias="llm_node"):
    """Unified LLM node — prompt building, API calls, and structured output.

    Capabilities
    ------------
    1. **Prompt fields**: ``system_message``, ``human_message`` — flat, explicit
    2. **Few-shot examples**: ``examples`` — appended to system message
    3. **Conversation history**: ``conversation`` — assembled from the DAG
    4. **Structured output**: ``output_schema`` — native (port) or fallback (middleware)
    5. **Backward compat**: ``prompt_template``, ``system_prompt``, ``parse_json`` still work

    Port Capabilities
    -----------------
    Requires ``llm`` port implementing ``SupportsGeneration`` (validated at mount time).
    The orchestrator auto-wraps ports with ``SupportsStructuredOutput`` middleware if needed.

    Examples
    --------
    New YAML spec::

        - kind: llm_node
          metadata:
            name: analyzer
          spec:
            system_message: "You are an expert analyst."
            human_message: "Analyze this: {{data}}"
            examples:
              - input: "Great product!"
                output: "positive"
            output_schema:
              sentiment: str
              confidence: float

    With conversation history::

        - kind: llm_node
          metadata:
            name: responder
          spec:
            system_message: "You are a helpful assistant."
            conversation:
              - prompt: "Recent conversation:"
                messages: "{{history_node}}"
            human_message: "{{user_input}}"
          dependencies: [history_node]

    Legacy YAML (still works, emits deprecation warnings)::

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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LLMNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        # --- New unified fields ---
        human_message: str | None = None,
        system_message: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        conversation: str | list[Any] | None = None,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        # --- Deprecated (backward compat) ---
        prompt_template: PromptInput | str | None = None,
        system_prompt: str | None = None,
        parse_json: bool = False,
        template: PromptInput | str | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a unified LLM node specification.

        Parameters
        ----------
        name : str
            Node name (must be unique in the graph)
        human_message : str | None
            User/human message template with ``{{variable}}`` syntax.
        system_message : str | None
            System message template.
        examples : list[dict[str, Any]] | None
            Few-shot examples (list of ``{input, output}`` dicts).
        conversation : str | list | None
            Conversation history. See class docstring for three forms.
        output_schema : dict[str, Any] | type[BaseModel] | None
            Structured output schema. Presence enables structured output.
        deps : list[str] | None
            List of dependency node names.
        prompt_template : str | None
            (Deprecated) Alias for ``human_message``.
        system_prompt : str | None
            (Deprecated) Alias for ``system_message``.
        parse_json : bool
            (Deprecated) Implied by presence of ``output_schema``.
        template : str | None
            (Deprecated) Alias for ``human_message``.

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution.
        """
        # --- Backward compatibility normalization ---
        actual_human = human_message
        actual_system = system_message

        # prompt_template / template → human_message
        legacy_template = prompt_template or template
        if legacy_template and not actual_human:
            if prompt_template:
                warnings.warn(
                    "LLMNode: 'prompt_template' is deprecated, use 'human_message'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if template:
                warnings.warn(
                    "LLMNode: 'template' is deprecated, use 'human_message'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            actual_human = legacy_template if isinstance(legacy_template, str) else None

        # system_prompt → system_message
        if system_prompt and not actual_system:
            warnings.warn(
                "LLMNode: 'system_prompt' is deprecated, use 'system_message'",
                DeprecationWarning,
                stacklevel=2,
            )
            actual_system = system_prompt

        # parse_json → implied by output_schema
        if parse_json:
            warnings.warn(
                "LLMNode: 'parse_json' is deprecated — set 'output_schema' instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # Determine if structured output is needed
        use_structured = output_schema is not None or parse_json

        if use_structured and output_schema is None:
            raise ValueError("output_schema is required for structured output")

        # --- Resolve template ---
        if actual_human is not None:
            prepared_template = self._prepare_template(actual_human)
        elif legacy_template is not None:
            prepared_template = self._prepare_template(legacy_template)
        else:
            raise ValueError("human_message (or prompt_template) is required")

        # Infer input schema from template variables + conversation refs
        input_schema = self.infer_input_schema_from_template(
            prepared_template,
            special_params={"context_history", "system_prompt"},
        )

        # Add conversation dependency variables to input schema
        if conversation:
            conv_vars = _extract_conversation_vars(conversation)
            for var in conv_vars:
                if var not in input_schema:
                    input_schema[var] = Any

        # Create output model if schema provided
        output_model: type[BaseModel] | None = None
        if output_schema is not None:
            output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Create the LLM wrapper function
        llm_wrapper = self._create_llm_wrapper(
            name=name,
            template=prepared_template,
            output_model=output_model,
            system_message=actual_system,
            examples=examples,
            conversation=conversation,
        )

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=llm_wrapper,
            input_schema=input_schema,
            output_schema=output_schema if use_structured else None,
            deps=deps,
            **kwargs,
        )

    # Template Processing
    # -------------------

    @staticmethod
    def _prepare_template(
        template: PromptInput | str | dict[str, Any],
    ) -> TemplateType:
        """Convert string or dict input to PromptTemplate if needed."""
        if isinstance(template, dict):
            return PromptTemplate.from_yaml(template)
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
        system_message: str | None,
        examples: list[dict[str, Any]] | None,
        conversation: str | list[Any] | None,
    ) -> Callable[..., Any]:
        """Create an async LLM wrapper function."""

        async def llm_wrapper(validated_input: dict[str, Any]) -> Any:
            """Execute LLM call with optional structured output."""
            node_logger = logger.bind(
                node=name,
                node_type="llm_node",
                pipeline_name=get_pipeline_name() or "unknown",
            )

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

                    node_logger.debug(
                        "Prompt variables",
                        variables=list(input_dict.keys()),
                        variable_count=len(input_dict),
                    )

                    node_logger.info(
                        "Calling LLM",
                        has_system_message=system_message is not None,
                        has_examples=examples is not None,
                        has_conversation=conversation is not None,
                        structured_output=output_model is not None,
                    )

                    # Build messages
                    messages = _build_messages(
                        template=template,
                        input_dict=input_dict,
                        system_message=system_message,
                        examples=examples,
                        conversation=conversation,
                    )

                    # Call LLM — structured or plain
                    if output_model:
                        result_dict = await llm.aresponse_structured(messages, output_model)
                        node_logger.debug(
                            "Structured response received",
                            duration_ms=t.duration_str,
                        )
                        return output_model.model_validate(result_dict)

                    response = await llm.aresponse(messages)
                    node_logger.debug(
                        "LLM response received",
                        response_length=(len(response) if isinstance(response, str) else None),
                        duration_ms=t.duration_str,
                    )
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
        """Create a NodeSpec from template (legacy compatibility method)."""
        return cls()(
            name=name,
            human_message=template if isinstance(template, str) else None,
            prompt_template=(template if not isinstance(template, str) else None),
            output_schema=output_schema,
            deps=deps,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_messages(
    template: TemplateType,
    input_dict: dict[str, Any],
    system_message: str | None,
    examples: list[dict[str, Any]] | None,
    conversation: str | list[Any] | None,
) -> list[Message]:
    """Build the full message array from the unified spec fields.

    Assembly order:
    1. system_message + examples
    2. conversation (resolved from inputs)
    3. human_message (rendered from template)
    """
    messages: list[Message] = []

    # 1. System message + examples
    if system_message:
        system_content = PromptTemplate(system_message).render(**input_dict)
        if examples:
            system_content += _format_examples(examples)
        messages.append(Message(role="system", content=system_content))
    elif examples:
        # Examples without system message — create a system message for them
        messages.append(Message(role="system", content=_format_examples(examples)))

    # 2. Conversation history
    if conversation:
        conv_messages = _resolve_conversation(conversation, input_dict)
        messages.extend(conv_messages)

    # 3. Human message (from template)
    message_dicts = template.to_messages(**input_dict)
    # Only take non-system messages from template output (system is handled above)
    messages.extend(
        Message.model_validate(msg_dict)
        for msg_dict in message_dicts
        if msg_dict.get("role") != "system"
    )

    # If template produced a system message and we don't have one yet, use it
    if not system_message and not examples:
        for msg_dict in message_dicts:
            if msg_dict.get("role") == "system":
                messages.insert(0, Message.model_validate(msg_dict))
                break

    return messages


def _extract_conversation_vars(
    conversation: str | list[Any],
) -> list[str]:
    """Extract variable names referenced in conversation field."""
    import re  # lazy: only needed for variable extraction

    pattern = r"\{\{\s*(\w+)\s*\}\}"
    vars_found: list[str] = []

    if isinstance(conversation, str):
        match = re.findall(pattern, conversation)
        vars_found.extend(match)
    elif isinstance(conversation, list):
        for item in conversation:
            if isinstance(item, dict) and isinstance(msg_ref := item.get("messages"), str):
                vars_found.extend(re.findall(pattern, msg_ref))

    return vars_found
