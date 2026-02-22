"""Builtin composable prompt templates for hexDAG.

This module provides reusable, registry-based prompt templates that can be
composed using the builder pattern:

    main_prompt + tool_prompt + fewshot_prompt

All prompts are registered in the component registry and can be referenced
in YAML configurations or composed programmatically.
"""

from hexdag.stdlib.prompts.base import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
)
from hexdag.stdlib.prompts.chat_prompts import (
    ChatAnalysisPrompt,
    ChatConversationalPrompt,
    ChatFewShotQAPrompt,
    ChatQAPrompt,
    FewShotClassificationPrompt,
    FewShotExtractionPrompt,
    create_chat_prompt,
    create_fewshot_prompt,
)
from hexdag.stdlib.prompts.error_correction_prompts import (
    GenericParseErrorPrompt,
    JsonParseErrorPrompt,
    JsonValidationErrorPrompt,
    MarkdownJsonErrorPrompt,
    SafeJsonInstructionsPrompt,
    get_error_correction_prompt,
)
from hexdag.stdlib.prompts.tool_prompts import (
    FunctionToolPrompt,
    JsonToolPrompt,
    MixedToolPrompt,
)

__all__ = [
    # Base template classes
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "ChatFewShotTemplate",
    # Tool prompts
    "FunctionToolPrompt",
    "JsonToolPrompt",
    "MixedToolPrompt",
    # Chat prompts
    "ChatQAPrompt",
    "ChatAnalysisPrompt",
    "ChatConversationalPrompt",
    "ChatFewShotQAPrompt",
    # Few-shot prompts
    "FewShotClassificationPrompt",
    "FewShotExtractionPrompt",
    # Error correction prompts
    "JsonParseErrorPrompt",
    "JsonValidationErrorPrompt",
    "MarkdownJsonErrorPrompt",
    "SafeJsonInstructionsPrompt",
    "GenericParseErrorPrompt",
    "get_error_correction_prompt",
    # Factories
    "create_chat_prompt",
    "create_fewshot_prompt",
]
