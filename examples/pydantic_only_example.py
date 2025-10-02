"""Example of the clean Pydantic-only approach for HexDAG.

This example demonstrates:
- NodeSpec with Pydantic models only
- Automatic validation by Pydantic
- No complex validation framework needed
- Clean separation of concerns
"""

import asyncio

from pydantic import BaseModel, Field

from hexai.core.application.orchestrator import Orchestrator

# Use the refactored imports with Pydantic support
from hexai.core.domain.dag import DirectedGraph, NodeSpec


# Define Pydantic models for data contracts
class TextInput(BaseModel):
    """Input text data."""

    text: str
    language: str = "en"
    max_length: int | None = None


class CleanedText(BaseModel):
    """Cleaned text with metadata."""

    original: str
    cleaned: str
    changes_made: list[str] = Field(default_factory=list)
    char_count: int


class TokenizedText(BaseModel):
    """Tokenized text data."""

    text: str
    tokens: list[str]
    token_count: int
    unique_tokens: set[str]


class AnalysisResult(BaseModel):
    """Final analysis result."""

    original_text: str
    cleaned_text: str
    tokens: list[str]
    sentiment_score: float
    complexity_score: float
    summary: str


# Define node functions that work with Pydantic models
async def load_text(input_data: None) -> TextInput:
    """Load text data."""
    return TextInput(
        text="  Hello WORLD!  This is a TEST message.  ", language="en", max_length=100
    )


async def clean_text(input_data: TextInput) -> CleanedText:
    """Clean the input text."""
    original = input_data.text
    cleaned = original.strip().lower()

    changes = []
    if original != original.strip():
        changes.append("trimmed whitespace")
    if original.lower() != original:
        changes.append("converted to lowercase")

    return CleanedText(
        original=original, cleaned=cleaned, changes_made=changes, char_count=len(cleaned)
    )


async def tokenize_text(input_data: CleanedText) -> TokenizedText:
    """Tokenize the cleaned text."""
    tokens = input_data.cleaned.split()

    return TokenizedText(
        text=input_data.cleaned, tokens=tokens, token_count=len(tokens), unique_tokens=set(tokens)
    )


async def analyze_text(input_data: TokenizedText) -> AnalysisResult:
    """Analyze the tokenized text."""
    # Simple mock analysis
    sentiment = 0.7 if "test" in input_data.tokens else 0.5
    complexity = len(input_data.unique_tokens) / len(input_data.tokens)

    return AnalysisResult(
        original_text=input_data.text,
        cleaned_text=input_data.text,
        tokens=input_data.tokens,
        sentiment_score=sentiment,
        complexity_score=complexity,
        summary=f"Analyzed {input_data.token_count} tokens",
    )


async def main():
    """Run example pipeline."""
    print("=" * 60)
    print("PYDANTIC-ONLY HEXDAG EXAMPLE")
    print("=" * 60)

    print("\nText Processing Pipeline")
    print("-" * 30)

    # Build DAG with Pydantic models
    graph = DirectedGraph()

    # Add nodes with Pydantic type contracts
    graph.add(NodeSpec(name="load", fn=load_text, out_model=TextInput))

    graph.add(
        NodeSpec(
            name="clean", fn=clean_text, in_model=TextInput, out_model=CleanedText, deps={"load"}
        )
    )

    graph.add(
        NodeSpec(
            name="tokenize",
            fn=tokenize_text,
            in_model=CleanedText,
            out_model=TokenizedText,
            deps={"clean"},
        )
    )

    graph.add(
        NodeSpec(
            name="analyze",
            fn=analyze_text,
            in_model=TokenizedText,
            out_model=AnalysisResult,
            deps={"tokenize"},
        )
    )

    # Execute with the refactored orchestrator
    orchestrator = Orchestrator(strict_validation=False)
    results = await orchestrator.run(graph, initial_input=None)

    # Display results
    for node_name, result in results.items():
        print(f"\n{node_name}:")
        if isinstance(result, BaseModel):
            print(f"  {result.model_dump_json(indent=2)}")
        else:
            print(f"  {result}")

    print("\n" + "=" * 60)
    print("BENEFITS OF THIS APPROACH:")
    print("- No complex validation framework")
    print("- Pydantic handles all type conversion")
    print("- Clean, explicit data contracts")
    print("- Excellent IDE support and documentation")
    print("- Simple and maintainable")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
