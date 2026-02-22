"""Base prompt template classes for building composable prompts.

Re-exported from their canonical location in the kernel. These classes
are available here for backward compatibility and convenience.
"""

from hexdag.kernel.orchestration.prompt.template import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    _extract_variables_cached,
)

__all__ = [
    "ChatFewShotTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "PromptTemplate",
    "_extract_variables_cached",
]
