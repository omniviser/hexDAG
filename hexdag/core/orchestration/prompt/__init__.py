"""Prompt template utilities for the application layer.

This module contains prompt template classes used by application nodes for rendering templated
prompts with variable substitution.
"""

from hexdag.core.orchestration.prompt.template import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)

# Type aliases for template types (used across the framework)
PromptInput = (
    str | PromptTemplate | ChatPromptTemplate | ChatFewShotTemplate | FewShotPromptTemplate
)
TemplateType = PromptTemplate | ChatPromptTemplate | ChatFewShotTemplate | FewShotPromptTemplate

__all__ = [
    "PromptTemplate",
    "FewShotPromptTemplate",
    "ChatPromptTemplate",
    "ChatFewShotTemplate",
    "PromptInput",
    "TemplateType",
]
