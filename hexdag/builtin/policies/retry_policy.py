"""Retry policies for automatic error recovery.

This module provides retry policies that work with the orchestrator
to automatically retry failed operations with improved inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hexdag.core.exceptions import ParseError
from hexdag.core.logging import get_logger

logger = get_logger(__name__)


class RetryContext(BaseModel):
    """Context information for retry attempts.

    Attributes
    ----------
    node_name : str
        Name of the node that failed
    attempt : int
        Current retry attempt (0 = first attempt)
    max_retries : int
        Maximum number of retries allowed
    error : str
        Error message from the failure
    original_input : dict
        Original input to the failed node
    previous_output : Any
        Output from previous node (if available)
    """

    node_name: str
    attempt: int
    max_retries: int
    error: str
    original_input: dict[str, Any]
    previous_output: Any | None = None

    model_config = {"arbitrary_types_allowed": True}


class ParseRetryPolicy:
    """Retry policy for parse failures in LLM workflows.

    When an LLMNode fails to parse LLM output, this policy:
    1. Detects the parse error
    2. Generates an improved prompt with error hints
    3. Re-executes the LLM with the improved prompt
    4. Repeats up to max_retries times

    This works at the **orchestrator level** - no need for dynamic graphs!

    Examples
    --------
    Basic usage::

        policy = ParseRetryPolicy(max_retries=2)

        # Orchestrator will automatically use this policy
        orchestrator = Orchestrator(
            ports={"llm": llm_adapter},
            policies=[policy]
        )

        # If parser fails, policy automatically retries with better prompt
        result = await orchestrator.aexecute(graph, inputs)

    Custom retry prompt builder::

        def custom_retry_prompt(ctx: RetryContext) -> str:
            return f'''
            Previous attempt failed: {ctx.error}

            Please try again with valid JSON format.
            Original prompt: {ctx.original_input.get("prompt")}
            '''

        policy = ParseRetryPolicy(
            max_retries=3,
            retry_prompt_builder=custom_retry_prompt
        )
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_prompt_builder: Any | None = None,
        backoff_multiplier: float = 1.5,
    ) -> None:
        """Initialize retry policy.

        Args
        ----
            max_retries: Maximum number of retry attempts
            retry_prompt_builder: Custom function to build retry prompts
            backoff_multiplier: Multiplier for exponential backoff (future use)
        """
        self.max_retries = max_retries
        self.retry_prompt_builder = retry_prompt_builder or self._default_retry_prompt
        self.backoff_multiplier = backoff_multiplier

    async def should_retry(self, error: Exception, context: RetryContext) -> bool:
        """Determine if we should retry this error.

        Args
        ----
            error: The exception that occurred
            context: Retry context with attempt info

        Returns
        -------
        bool
            True if we should retry
        """
        # Only retry ParseErrors
        if not isinstance(error, ParseError):
            return False

        # Check if we have retries left
        if context.attempt >= self.max_retries:
            logger.warning(
                f"Max retries ({self.max_retries}) reached for node '{context.node_name}'"
            )
            return False

        logger.info(
            f"Retry attempt {context.attempt + 1}/{self.max_retries} for node '{context.node_name}'"
        )
        return True

    async def prepare_retry_input(self, context: RetryContext) -> dict[str, Any]:
        """Prepare improved input for retry attempt.

        Args
        ----
            context: Retry context with error info

        Returns
        -------
        dict[str, Any]
            Modified input for retry
        """
        # Build improved prompt with error hints
        improved_prompt = self.retry_prompt_builder(context)

        # Create new input with improved prompt
        retry_input = context.original_input.copy()

        # Update the prompt/template in the input
        if "template" in retry_input:
            retry_input["template"] = improved_prompt
        elif "prompt" in retry_input:
            retry_input["prompt"] = improved_prompt
        elif "text" in retry_input:
            retry_input["text"] = improved_prompt
        else:
            # Fallback: add as a new field
            retry_input["retry_prompt"] = improved_prompt

        logger.debug(f"Prepared retry input with improved prompt (attempt {context.attempt + 1})")

        return retry_input

    def _default_retry_prompt(self, context: RetryContext) -> str:
        """Build default retry prompt using registered error correction templates.

        Args
        ----
            context: Retry context

        Returns
        -------
        str
            Improved prompt text using error correction templates
        """
        from hexdag.builtin.prompts import get_error_correction_prompt

        original_prompt = context.original_input.get("prompt", "")
        if not original_prompt:
            original_prompt = context.original_input.get("template", "")
        if not original_prompt:
            original_prompt = context.original_input.get("text", "")

        # Determine error type
        error_type = self._classify_error(context.error)

        # Get appropriate error correction prompt from registry
        ErrorPromptClass = get_error_correction_prompt(error_type, strategy="json")
        error_prompt = ErrorPromptClass()  # type: ignore[call-arg]

        # Render the error correction prompt
        return error_prompt.render(
            original_prompt=original_prompt,
            llm_output=context.previous_output or "",
            error_message=self._extract_error_summary(context.error),
            schema=str(context.original_input.get("output_schema", {})),
        )

    def _classify_error(self, error_msg: str) -> str:
        """Classify the type of parse error.

        Args
        ----
            error_msg: Error message

        Returns
        -------
        str
            Error type: "parse", "validation", "markdown", or "generic"
        """
        error_lower = error_msg.lower()

        # Check for specific error patterns
        if "json" in error_lower and any(
            keyword in error_lower
            for keyword in ["expecting", "invalid", "syntax", "decode", "malformed"]
        ):
            return "parse"

        if any(keyword in error_lower for keyword in ["validation", "schema", "required field"]):
            return "validation"

        if any(keyword in error_lower for keyword in ["markdown", "code block", "```"]):
            return "markdown"

        return "generic"

    def _extract_error_summary(self, error_msg: str) -> str:
        """Extract concise error summary from full error message.

        Args
        ----
            error_msg: Full error message

        Returns
        -------
        str
            Concise error summary
        """
        # Extract first few lines (usually most relevant)
        lines = error_msg.split("\n")
        summary_lines = []

        for line in lines[:10]:  # First 10 lines
            if "Retry hints" in line:
                break  # Stop before retry hints section
            if line.strip():
                summary_lines.append(line.strip())

        summary = "\n".join(summary_lines)

        # Truncate if too long
        if len(summary) > 500:
            summary = summary[:500] + "..."

        return summary


class ExponentialBackoffPolicy:
    """Exponential backoff for rate limit errors.

    Implements exponential backoff with jitter for retrying
    rate-limited API calls.

    Examples
    --------
        policy = ExponentialBackoffPolicy(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0
        )
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
    ) -> None:
        """Initialize exponential backoff policy.

        Args
        ----
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier

    async def should_retry(self, error: Exception, context: RetryContext) -> bool:
        """Check if error is retriable."""
        # Check for rate limit errors
        error_str = str(error).lower()
        is_rate_limit = any(
            keyword in error_str for keyword in ["rate limit", "too many requests", "429"]
        )

        return is_rate_limit and context.attempt < self.max_retries

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt.

        Args
        ----
            attempt: Retry attempt number

        Returns
        -------
        float
            Delay in seconds
        """
        return min(self.base_delay * (self.multiplier**attempt), self.max_delay)
