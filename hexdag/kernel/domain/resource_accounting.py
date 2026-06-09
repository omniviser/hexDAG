"""Resource accounting domain models.

Tracks cumulative resource consumption (tokens, API calls, duration)
during pipeline execution.  Used by the ``ResourceAccounting`` middleware
to enforce per-pipeline limits and emit ``ResourceWarning`` /
``ResourceLimitExceeded`` events.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResourceUsage:
    """Cumulative resource consumption for a pipeline run.

    Attributes
    ----------
    total_tokens : int
        Total LLM tokens consumed (input + output).
    input_tokens : int
        Total LLM input tokens consumed.
    output_tokens : int
        Total LLM output tokens consumed.
    llm_calls : int
        Number of LLM API calls made.
    tool_calls : int
        Number of tool router calls made.
    total_duration_ms : float
        Cumulative wall-clock duration of all port calls.
    """

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    total_duration_ms: float = 0.0

    def add_llm_call(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an LLM call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.llm_calls += 1
        self.total_duration_ms += duration_ms

    def add_tool_call(self, *, duration_ms: float = 0.0) -> None:
        """Record a tool call."""
        self.tool_calls += 1
        self.total_duration_ms += duration_ms


@dataclass(slots=True)
class ResourceLimits:
    """Per-pipeline resource limits.

    Any field set to ``None`` means no limit for that resource.

    Attributes
    ----------
    max_total_tokens : int | None
        Maximum total LLM tokens.
    max_llm_calls : int | None
        Maximum number of LLM API calls.
    max_tool_calls : int | None
        Maximum number of tool router calls.
    max_duration_ms : float | None
        Maximum cumulative port call duration in ms.
    warning_threshold : float
        Fraction (0.0–1.0) at which to emit a ``ResourceWarning``.
        Default 0.8 (80% of limit).
    """

    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_duration_ms: float | None = None
    warning_threshold: float = 0.8

    def check(self, usage: ResourceUsage) -> list[LimitCheck]:
        """Check usage against all configured limits.

        Returns
        -------
        list[LimitCheck]
            One entry per configured limit, with status and details.
        """
        checks: list[LimitCheck] = []
        wt = self.warning_threshold
        if self.max_total_tokens is not None:
            checks.append(_check_one("total_tokens", usage.total_tokens, self.max_total_tokens, wt))
        if self.max_llm_calls is not None:
            checks.append(_check_one("llm_calls", usage.llm_calls, self.max_llm_calls, wt))
        if self.max_tool_calls is not None:
            checks.append(_check_one("tool_calls", usage.tool_calls, self.max_tool_calls, wt))
        if self.max_duration_ms is not None:
            checks.append(
                _check_one("duration_ms", usage.total_duration_ms, self.max_duration_ms, wt)
            )
        return checks


@dataclass(slots=True)
class LimitCheck:
    """Result of checking one resource against its limit.

    Attributes
    ----------
    resource : str
        Resource name (e.g., ``"total_tokens"``).
    current : float
        Current usage value.
    limit : float
        Configured limit.
    ratio : float
        Usage as a fraction of limit (0.0–1.0+).
    status : str
        ``"ok"``, ``"warning"``, or ``"exceeded"``.
    """

    resource: str
    current: float
    limit: float
    ratio: float
    status: str  # "ok" | "warning" | "exceeded"


def _check_one(
    resource: str,
    current: float,
    limit: float,
    warning_threshold: float,
) -> LimitCheck:
    """Check a single resource value against its limit."""
    ratio = current / limit if limit > 0 else 0.0
    if ratio >= 1.0:
        status = "exceeded"
    elif ratio >= warning_threshold:
        status = "warning"
    else:
        status = "ok"
    return LimitCheck(
        resource=resource,
        current=current,
        limit=limit,
        ratio=ratio,
        status=status,
    )
