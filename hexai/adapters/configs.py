"""Pydantic configuration models for mock adapters."""

from pydantic import BaseModel, Field


class MockLLMConfig(BaseModel):
    """Configuration for MockLLM adapter."""

    responses: list[str] | str | None = Field(
        default=None,
        description="Single response string, list of responses, or None for default",
    )
    delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Artificial delay to simulate API latency",
    )


class MockDatabaseConfig(BaseModel):
    """Configuration for MockDatabase adapter."""

    enable_sample_data: bool = Field(
        default=True,
        description="Whether to initialize with sample database schema",
    )
    delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Artificial delay to simulate database query latency",
    )


class InMemoryMemoryConfig(BaseModel):
    """Configuration for InMemoryMemory adapter."""

    delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Artificial delay to simulate storage access latency",
    )
    max_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of items to store (None for unlimited)",
    )


class MockToolRouterConfig(BaseModel):
    """Configuration for MockToolRouter adapter."""

    available_tools: list[str] = Field(
        default_factory=list,
        description="List of available tool names",
    )
    delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Artificial delay to simulate tool execution",
    )
    raise_on_unknown_tool: bool = Field(
        default=True,
        description="Whether to raise an error for unknown tools",
    )
