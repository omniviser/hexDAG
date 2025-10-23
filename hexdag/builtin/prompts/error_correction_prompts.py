"""Error correction and retry prompt templates.

These prompts help LLMs fix their mistakes when parsing fails.
They provide clear instructions on what went wrong and how to fix it.
"""

from hexdag.core.orchestration.prompt.template import PromptTemplate
from hexdag.core.registry import prompt


@prompt(
    name="json_parse_error",
    namespace="core",
    description="Corrective prompt for JSON parse errors",
)
class JsonParseErrorPrompt(PromptTemplate):
    """Prompt to help fix JSON parsing errors.

    Uses clear examples and common mistake patterns to guide the LLM
    toward producing valid JSON.

    Variables
    ---------
    - original_prompt: The original prompt that was given
    - llm_output: The output that failed to parse
    - error_message: The parse error message
    - schema: Expected JSON schema (optional)

    Examples
    --------
    Usage in ParseRetryPolicy::

        retry_prompt = JsonParseErrorPrompt()
        corrected = retry_prompt.render(
            original_prompt="Generate user data",
            llm_output='name: "John", age: 30',  # Invalid JSON
            error_message="Expecting property name enclosed in double quotes",
            schema={"name": "str", "age": "int"}
        )
    """

    def __init__(self) -> None:
        """Initialize JSON parse error correction prompt."""
        template = """{{original_prompt}}

⚠️ PREVIOUS ATTEMPT FAILED - JSON PARSING ERROR

Your previous response could not be parsed as valid JSON:
```
{{llm_output}}
```

Error: {{error_message}}

IMPORTANT: Respond with VALID JSON that follows these rules:

1. ✅ Use double quotes for strings: "text" not 'text'
2. ✅ No trailing commas: {"a": 1, "b": 2} not {"a": 1, "b": 2,}
3. ✅ Property names must be quoted: {"name": "value"} not {name: "value"}
4. ✅ Use true/false not True/False
5. ✅ Use null not None

Common mistakes to avoid:
- ❌ Single quotes: {'name': 'John'}
- ❌ Trailing commas: {"age": 30,}
- ❌ Unquoted keys: {name: "John"}
- ❌ Python booleans: {"active": True}

Correct format:
- ✅ {"name": "John", "age": 30, "active": true}

{{#schema}}Expected schema:
```json
{{schema}}
```{{/schema}}

Please respond ONLY with valid JSON. No explanations, no markdown formatting.
"""
        super().__init__(template)


@prompt(
    name="json_validation_error",
    namespace="core",
    description="Corrective prompt for JSON schema validation errors",
)
class JsonValidationErrorPrompt(PromptTemplate):
    """Prompt to fix schema validation errors.

    When JSON parses correctly but doesn't match the expected schema,
    this prompt helps the LLM understand what fields are missing or incorrect.

    Variables
    ---------
    - original_prompt: The original prompt
    - llm_output: The parsed JSON that failed validation
    - validation_errors: List of validation errors
    - required_fields: Required fields in the schema
    - schema: Full schema specification

    Examples
    --------
    Usage::

        retry_prompt = JsonValidationErrorPrompt()
        corrected = retry_prompt.render(
            original_prompt="Generate user profile",
            llm_output='{"name": "John"}',  # Missing 'age' field
            validation_errors=["Field 'age' required"],
            required_fields=["name", "age", "email"],
            schema={"name": "str", "age": "int", "email": "str"}
        )
    """

    def __init__(self) -> None:
        """Initialize JSON validation error correction prompt."""
        template = """{{original_prompt}}

⚠️ PREVIOUS ATTEMPT FAILED - SCHEMA VALIDATION ERROR

Your response was valid JSON but did NOT match the required schema.

Your output:
```json
{{llm_output}}
```

Validation errors:
{{validation_errors}}

Required schema:
```json
{{schema}}
```

REQUIRED FIELDS (must include ALL):
{{#required_fields}}- {{.}}
{{/required_fields}}

Instructions:
1. Include ALL required fields
2. Use correct data types (string, number, boolean, array, object)
3. Field names must match EXACTLY (case-sensitive)
4. Arrays must contain the correct element types

Example of CORRECT format:
```json
{
  "name": "John Doe",
  "age": 30,
  "email": "john@example.com",
  "active": true
}
```

Please respond with valid JSON that matches the schema exactly.
"""
        super().__init__(template)


@prompt(
    name="safe_json_instructions",
    namespace="core",
    description="Proactive JSON formatting instructions to prevent errors",
)
class SafeJsonInstructionsPrompt(PromptTemplate):
    """Proactive JSON instructions to prevent common errors.

    Add this to prompts that expect JSON output to reduce errors upfront.

    Variables
    ---------
    - schema: Expected JSON schema

    Examples
    --------
    Compose with main prompt::

        from hexdag.builtin.prompts import SafeJsonInstructionsPrompt

        main_prompt = PromptTemplate("Generate user data: {{requirements}}")
        json_instructions = SafeJsonInstructionsPrompt()

        full_prompt = main_prompt + json_instructions
    """

    def __init__(self) -> None:
        """Initialize safe JSON instructions prompt."""
        template = """
## JSON Output Requirements

IMPORTANT: Your response must be valid JSON following these rules:

✅ Valid JSON format:
- Use double quotes for strings: "text"
- No trailing commas
- Quote all property names
- Use true/false (not True/False)
- Use null (not None)

{{#schema}}Expected structure:
```json
{{schema}}
```{{/schema}}

Respond with ONLY the JSON object. No explanations before or after.
"""
        super().__init__(template)


@prompt(
    name="markdown_json_error",
    namespace="core",
    description="Corrective prompt when JSON is wrapped in markdown",
)
class MarkdownJsonErrorPrompt(PromptTemplate):
    """Prompt to fix JSON wrapped in markdown code blocks.

    Helps when the LLM correctly generates JSON but wraps it in
    markdown formatting that breaks parsing.

    Variables
    ---------
    - original_prompt: Original prompt
    - error_message: Parse error

    Examples
    --------
        retry_prompt = MarkdownJsonErrorPrompt()
        corrected = retry_prompt.render(
            original_prompt="Generate config",
            error_message="Could not find JSON in response"
        )
    """

    def __init__(self) -> None:
        """Initialize markdown JSON error prompt."""
        template = """{{original_prompt}}

⚠️ PREVIOUS ATTEMPT FAILED - JSON FORMATTING ERROR

Error: {{error_message}}

IMPORTANT: Respond with ONLY the raw JSON object.

❌ DO NOT wrap in markdown code blocks:
```json
{"name": "value"}
```

❌ DO NOT add explanations:
Here is the JSON: {"name": "value"}

✅ CORRECT - Just the JSON:
{"name": "value"}

Please provide ONLY the JSON object, nothing else.
"""
        super().__init__(template)


@prompt(
    name="generic_parse_error",
    namespace="core",
    description="Generic error correction prompt for parse failures",
)
class GenericParseErrorPrompt(PromptTemplate):
    """Generic parse error correction prompt.

    Fallback for parse errors that don't fit specific categories.

    Variables
    ---------
    - original_prompt: Original prompt
    - llm_output: Failed output
    - error_message: Error message
    - format: Expected format (json, yaml, etc.)

    Examples
    --------
        retry_prompt = GenericParseErrorPrompt()
        corrected = retry_prompt.render(
            original_prompt="Extract data",
            llm_output="invalid data...",
            error_message="Parse failed",
            format="json"
        )
    """

    def __init__(self) -> None:
        """Initialize generic parse error prompt."""
        template = """{{original_prompt}}

⚠️ PREVIOUS ATTEMPT FAILED

Your previous response could not be parsed:
```
{{llm_output}}
```

Error: {{error_message}}

Expected format: {{format}}

Please provide a response in valid {{format}} format.
Ensure:
- Correct syntax
- Proper formatting
- All required fields included

Try again with a correctly formatted response.
"""
        super().__init__(template)


def get_error_correction_prompt(error_type: str, strategy: str = "json") -> type[PromptTemplate]:
    """Get appropriate error correction prompt for error type.

    Args
    ----
        error_type: Type of error ("parse", "validation", "markdown", "generic")
        strategy: Parsing strategy ("json", "yaml", etc.)

    Returns
    -------
    type[PromptTemplate]
        Appropriate error correction prompt class

    Examples
    --------
        PromptClass = get_error_correction_prompt("parse", "json")
        prompt = PromptClass()
        corrected = prompt.render(original_prompt="...", error_message="...")
    """
    if error_type == "parse":
        if strategy == "json":
            return JsonParseErrorPrompt
        return GenericParseErrorPrompt

    if error_type == "validation":
        return JsonValidationErrorPrompt

    if error_type == "markdown":
        return MarkdownJsonErrorPrompt

    return GenericParseErrorPrompt
