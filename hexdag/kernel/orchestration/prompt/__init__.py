"""Prompt template utilities for the application layer.

This module contains prompt template classes used by application nodes for rendering templated
prompts with variable substitution.

Note: ChatPromptTemplate, FewShotPromptTemplate, and ChatFewShotTemplate have been moved to
hexdag.stdlib.prompts.base but are re-exported here for backward compatibility.
"""

# Core template (minimal)
# Advanced templates (from builtin, re-exported for backward compatibility)
from hexdag.kernel.orchestration.prompt.template import PromptTemplate
from hexdag.stdlib.prompts.base import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
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
