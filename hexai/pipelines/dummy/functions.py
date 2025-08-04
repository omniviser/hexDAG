"""Functions for the dummy pipeline demonstrating hexAI framework capabilities."""

from typing import Any

from pydantic import BaseModel, Field


class DummyPipelineInput(BaseModel):
    """Input model for the dummy pipeline with proper type definitions."""

    text: str = Field(..., description="Text to analyze and validate")
    priority: int = Field(default=1, description="Priority level (1-10)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ValidatedOutput(BaseModel):
    """Output from input validation with typed fields."""

    text: str = Field(..., description="Validated text")
    priority: int = Field(..., description="Validated priority")
    metadata: dict[str, Any] = Field(..., description="Validated metadata")
    validated: bool = Field(..., description="Validation status")


class FeaturesOutput(BaseModel):
    """Output from feature extraction with typed fields."""

    word_count: int = Field(..., description="Number of words")
    char_count: int = Field(..., description="Number of characters")
    sentence_count: int = Field(..., description="Number of sentences")
    has_numbers: bool = Field(..., description="Contains numbers")
    has_uppercase: bool = Field(..., description="Contains uppercase letters")
    avg_word_length: float = Field(..., description="Average word length")
    original_data: dict[str, Any] = Field(..., description="Original input data")


class ScoreOutput(BaseModel):
    """Output from score calculation with typed fields."""

    quality_score: float = Field(..., description="Overall quality score")
    score_breakdown: dict[str, Any] = Field(..., description="Detailed score breakdown")
    features: dict[str, Any] = Field(..., description="Feature analysis")


class SummaryOutput(BaseModel):
    """Final pipeline output with comprehensive analysis."""

    analysis_summary: dict[str, Any] = Field(..., description="Complete analysis summary")
    recommendations: list[str] = Field(..., description="Recommendations for improvement")
    score_breakdown: dict[str, Any] = Field(..., description="Quality score details")
    metadata: dict[str, Any] = Field(..., description="Processing metadata")


async def validate_input(input_data: DummyPipelineInput, **ports: Any) -> ValidatedOutput:
    """Validate and prepare input data using Pydantic.

    Args
    ----
        input_data: Input data with text, priority, and metadata
        **ports: Injected ports (not used in this function)

    Returns
    -------
        ValidatedOutput with validated input data
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("validate_input", "Validating input data...")

    # Pydantic handles validation automatically
    text = input_data.text
    priority = input_data.priority

    # Additional business logic validation
    if not text or len(text.strip()) < 5:
        raise ValueError("Text must be at least 5 characters long")

    if priority < 1 or priority > 10:
        raise ValueError("Priority must be between 1 and 10")

    if event_manager:
        event_manager.add_trace("validate_input", f"Validated text with {len(text)} characters")

    return ValidatedOutput(
        text=text.strip(),
        priority=priority,
        metadata=input_data.metadata,
        validated=True,
    )


async def extract_features(input_data: ValidatedOutput, **ports: Any) -> FeaturesOutput:
    """Extract basic features from the validated text.

    Args
    ----
        input_data: Validated input data
        **ports: Injected ports (not used in this function)

    Returns
    -------
        FeaturesOutput with extracted features
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("extract_features", "Extracting text features...")

    # Use validated data directly
    text = input_data.text

    # Extract features
    words = text.split()
    sentences = text.split(".")

    # Count features
    word_count = len(words)
    char_count = len(text)
    sentence_count = len([s for s in sentences if s.strip()])
    has_numbers = any(char.isdigit() for char in text)
    has_uppercase = any(char.isupper() for char in text)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    if event_manager:
        event_manager.add_trace(
            "extract_features", f"Extracted {word_count} words, {char_count} characters"
        )

    return FeaturesOutput(
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        has_numbers=has_numbers,
        has_uppercase=has_uppercase,
        avg_word_length=round(avg_word_length, 2),
        original_data=input_data.model_dump(),
    )


async def calculate_score(input_data: FeaturesOutput, **ports: Any) -> ScoreOutput:
    """Calculate quality score based on extracted features.

    Args
    ----
        input_data: Features extracted from text
        **ports: Injected ports (not used in this function)

    Returns
    -------
        ScoreOutput with calculated quality score
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("calculate_score", "Calculating quality score...")

    # Use features directly
    word_count = input_data.word_count
    char_count = input_data.char_count
    sentence_count = input_data.sentence_count
    has_numbers = input_data.has_numbers
    has_uppercase = input_data.has_uppercase
    avg_word_length = input_data.avg_word_length

    # Calculate quality score based on features
    score = 0.0
    breakdown = {}

    # Word count score (0-25 points)
    if word_count >= 10:
        score += 25
        breakdown["word_count"] = 25
    elif word_count >= 5:
        score += 15
        breakdown["word_count"] = 15
    else:
        breakdown["word_count"] = 0

    # Character count score (0-20 points)
    if char_count >= 50:
        score += 20
        breakdown["char_count"] = 20
    elif char_count >= 25:
        score += 10
        breakdown["char_count"] = 10
    else:
        breakdown["char_count"] = 0

    # Sentence structure score (0-20 points)
    if sentence_count >= 2:
        score += 20
        breakdown["sentence_structure"] = 20
    elif sentence_count >= 1:
        score += 10
        breakdown["sentence_structure"] = 10
    else:
        breakdown["sentence_structure"] = 0

    # Content variety score (0-15 points)
    variety_score = 0
    if has_numbers:
        variety_score += 5
    if has_uppercase:
        variety_score += 5
    if avg_word_length >= 4:
        variety_score += 5
    score += variety_score
    breakdown["content_variety"] = variety_score

    # Readability score (0-20 points)
    if avg_word_length >= 3 and avg_word_length <= 8:
        score += 20
        breakdown["readability"] = 20
    elif avg_word_length >= 2 and avg_word_length <= 10:
        score += 10
        breakdown["readability"] = 10
    else:
        breakdown["readability"] = 0

    if event_manager:
        event_manager.add_trace("calculate_score", f"Calculated quality score: {score}/100")

    return ScoreOutput(
        quality_score=round(score, 2),
        score_breakdown=breakdown,
        features=input_data.model_dump(),
    )


async def generate_summary(input_data: ScoreOutput, **ports: Any) -> SummaryOutput:
    """Generate comprehensive summary from score analysis.

    Args
    ----
        input_data: Score analysis results
        **ports: Injected ports (not used in this function)

    Returns
    -------
        SummaryOutput with comprehensive analysis
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("generate_summary", "Generating comprehensive summary...")

    # Use score data directly
    quality_score = input_data.quality_score
    score_breakdown = input_data.score_breakdown
    features = input_data.features

    # Generate analysis summary
    analysis_summary: dict[str, Any] = {
        "overall_score": quality_score,
        "score_category": (
            "Excellent"
            if quality_score >= 80
            else "Good" if quality_score >= 60 else "Fair" if quality_score >= 40 else "Poor"
        ),
        "strengths": [],
        "weaknesses": [],
        "feature_analysis": features,
    }

    # Identify strengths and weaknesses
    if score_breakdown.get("word_count", 0) >= 15:
        analysis_summary["strengths"].append("Good word count")
    else:
        analysis_summary["weaknesses"].append("Low word count")

    if score_breakdown.get("char_count", 0) >= 10:
        analysis_summary["strengths"].append("Adequate character count")
    else:
        analysis_summary["weaknesses"].append("Too short")

    if score_breakdown.get("sentence_structure", 0) >= 10:
        analysis_summary["strengths"].append("Good sentence structure")
    else:
        analysis_summary["weaknesses"].append("Poor sentence structure")

    if score_breakdown.get("content_variety", 0) >= 10:
        analysis_summary["strengths"].append("Good content variety")
    else:
        analysis_summary["weaknesses"].append("Limited content variety")

    if score_breakdown.get("readability", 0) >= 10:
        analysis_summary["strengths"].append("Good readability")
    else:
        analysis_summary["weaknesses"].append("Poor readability")

    # Generate recommendations
    recommendations = []
    if quality_score < 60:
        recommendations.append("Consider adding more content to improve word count")
    if score_breakdown.get("sentence_structure", 0) < 10:
        recommendations.append("Add more sentences to improve structure")
    if score_breakdown.get("content_variety", 0) < 10:
        recommendations.append("Include numbers and varied case for better variety")
    if score_breakdown.get("readability", 0) < 10:
        recommendations.append("Adjust word length for better readability")

    if event_manager:
        event_manager.add_trace(
            "generate_summary", f"Generated summary with {len(recommendations)} recommendations"
        )

    return SummaryOutput(
        analysis_summary=analysis_summary,
        recommendations=recommendations,
        score_breakdown=score_breakdown,
        metadata={"node_name": "generate_summary", "quality_score": quality_score},
    )
