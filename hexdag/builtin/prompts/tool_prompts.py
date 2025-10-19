"""Composable tool usage prompt templates.

These prompts provide tool calling instructions that can be composed
with other prompts using the builder pattern.
"""

from hexdag.builtin.nodes.tool_utils import ToolCallFormat
from hexdag.core.orchestration.prompt.template import PromptTemplate
from hexdag.core.registry import prompt


@prompt(
    name="tool_usage_function", namespace="core", description="Function-style tool calling prompt"
)
class FunctionToolPrompt(PromptTemplate):
    """Tool usage prompt for function-style tool calls.

    Format: INVOKE_TOOL: tool_name(param='value')

    This prompt is composable - add it to any base prompt:
        main_prompt + FunctionToolPrompt()

    Examples
    --------
    Standalone usage:
        >>> prompt = FunctionToolPrompt()
        >>> text = prompt.render(
        ...     tools="search(query), calculator(expression)",
        ...     tool_schemas=[...]
        ... )

    Composed with main prompt:
        >>> main = PromptTemplate("Analyze {{data}}")
        >>> full_prompt = main + FunctionToolPrompt()
    """

    def __init__(self) -> None:
        """Initialize function-style tool prompt template."""
        template = """
## Available Tools
{{tools}}

## Usage Guidelines
- Call ONE tool at a time using: INVOKE_TOOL: tool_name(param='value')
- Use single quotes for string parameters
- For final answer and structured output: INVOKE_TOOL: tool_end(field1='value1', field2='value2')
- For phase change: INVOKE_TOOL: change_phase(phase='new_phase', reason='why changing', carried_data={'key': 'value'})

## Examples
INVOKE_TOOL: search(query='latest AI news')
INVOKE_TOOL: calculator(expression='2 + 2')
INVOKE_TOOL: tool_end(result='Analysis complete', confidence='high')
"""
        super().__init__(template, input_vars=["tools"])


@prompt(name="tool_usage_json", namespace="core", description="JSON-style tool calling prompt")
class JsonToolPrompt(PromptTemplate):
    """Tool usage prompt for JSON-style tool calls.

    Format: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}

    This prompt is composable - add it to any base prompt:
        main_prompt + JsonToolPrompt()

    Examples
    --------
    Standalone usage:
        >>> prompt = JsonToolPrompt()
        >>> text = prompt.render(tools="search, calculator", tool_schemas=[...])

    Composed with main prompt:
        >>> main = PromptTemplate("Analyze {{data}}")
        >>> full_prompt = main + JsonToolPrompt()
    """

    def __init__(self) -> None:
        """Initialize JSON-style tool prompt template."""
        template = """
## Available Tools
{{tools}}

## Usage Guidelines
- Call ONE tool at a time using JSON format
- Format: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}
- Use valid JSON syntax with double quotes
- For final answer: INVOKE_TOOL: {"tool": "tool_end", "params": {"field1": "value1", "field2": "value2"}}
- For phase change: INVOKE_TOOL: {"tool": "change_phase", "params": {"phase": "new_phase", "reason": "why", "carried_data": {"key": "val"}}}

## Examples
INVOKE_TOOL: {"tool": "search", "params": {"query": "latest AI news"}}
INVOKE_TOOL: {"tool": "calculator", "params": {"expression": "2 + 2"}}
INVOKE_TOOL: {"tool": "tool_end", "params": {"result": "Analysis complete", "confidence": "high"}}
"""
        super().__init__(template, input_vars=["tools"])


@prompt(
    name="tool_usage_mixed",
    namespace="core",
    description="Mixed-style tool calling prompt (function or JSON)",
)
class MixedToolPrompt(PromptTemplate):
    """Tool usage prompt supporting both function and JSON styles.

    Format: Either function or JSON style
    - Function: INVOKE_TOOL: tool_name(param='value')
    - JSON: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}

    This prompt is composable - add it to any base prompt:
        main_prompt + MixedToolPrompt()

    Examples
    --------
    Standalone usage:
        >>> prompt = MixedToolPrompt()
        >>> text = prompt.render(tools="search, calculator", tool_schemas=[...])

    Composed with main prompt:
        >>> main = PromptTemplate("Analyze {{data}}")
        >>> full_prompt = main + MixedToolPrompt()
    """

    def __init__(self) -> None:
        """Initialize mixed-style tool prompt template."""
        template = """
## Available Tools
{{tools}}

## Usage Guidelines
- Call ONE tool at a time using either format:
  - Function style: INVOKE_TOOL: tool_name(param='value')
  - JSON style: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}

### Function Style Examples
INVOKE_TOOL: search(query='latest AI news')
INVOKE_TOOL: tool_end(result='Analysis complete', confidence='high')
INVOKE_TOOL: change_phase(phase='refinement', reason='need more data', carried_data={'initial': 'findings'})

### JSON Style Examples
INVOKE_TOOL: {"tool": "search", "params": {"query": "latest AI news"}}
INVOKE_TOOL: {"tool": "tool_end", "params": {"result": "Analysis complete", "confidence": "high"}}
INVOKE_TOOL: {"tool": "change_phase", "params": {"phase": "refinement", "reason": "need more data", "carried_data": {"initial": "findings"}}}
"""
        super().__init__(template, input_vars=["tools"])


def get_tool_prompt_for_format(format_style: ToolCallFormat) -> type[PromptTemplate]:
    """Get the appropriate tool prompt class for a given format.

    Args
    ----
        format_style: Tool call format (FUNCTION_CALL, JSON, or MIXED)

    Returns
    -------
    type[PromptTemplate]
        Appropriate tool prompt class

    Examples
    --------
        >>> PromptClass = get_tool_prompt_for_format(ToolCallFormat.FUNCTION_CALL)
        >>> prompt = PromptClass()
        >>> text = prompt.render(tools="search, calculator")
    """
    if format_style == ToolCallFormat.FUNCTION_CALL:
        return FunctionToolPrompt
    if format_style == ToolCallFormat.JSON:
        return JsonToolPrompt
    return MixedToolPrompt
