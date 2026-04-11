"""Pydantic schema models for insight extraction output.

This module defines the contract (SPEC.md section 3) for all insight extraction outputs.
All LLM responses must validate against these models.
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """Metadata about the input source (YouTube URL or local file)."""

    type: Literal["youtube", "local_video", "local_audio"]
    url_or_path: str = Field(..., description="YouTube URL or file path")
    title: str | None = Field(None, description="Video/audio title if available")
    duration_seconds: int = Field(..., description="Total duration in seconds")


class Decision(BaseModel):
    """Binary decision and confidence level for watching the full content."""

    watch_full: bool = Field(..., description="Should the viewer watch the entire video?")
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence level of the decision"
    )
    rationale: str = Field(
        ..., max_length=280, description="Reason for decision (fits in a tweet)"
    )


class KeyConcept(BaseModel):
    """A single key concept extracted from the video."""

    name: str = Field(..., max_length=60, description="Concept name")
    explanation: str = Field(
        ..., max_length=240, description="Clear, technical explanation"
    )
    timestamp_seconds: int | None = Field(
        None, description="Optional timestamp where concept appears"
    )


class Metadata(BaseModel):
    """Processing metadata for debugging and benchmarking."""

    extracted_at: datetime = Field(..., description="When extraction occurred")
    transcription_model: str = Field(..., description="e.g., 'whisper-base'")
    llm_model: str = Field(..., description="e.g., 'qwen2.5:7b'")
    transcription_duration_seconds: float = Field(..., description="Time to transcribe")
    llm_duration_seconds: float = Field(..., description="Time for LLM to process")
    language_detected: str = Field(
        ..., description="ISO 639-1 language code (e.g., 'pt', 'en')"
    )


class Insight(BaseModel):
    """Complete insight extraction output — the root schema (SPEC.md section 3).

    v1.1.0 — schema generalized to work on any video topic (not just technical
    content), while keeping the reader framed as an IT professional reading
    critically. Renames and new fields are driven by reading lenses:

    - summary / notable_quotes → "shape the content before the detail"
    - core_thesis              → "separate the thesis from its illustrations"
    - caveats                  → "what was not said / not supported"
    - open_questions           → "what the video leaves open"
    """

    schema_version: Literal["1.1.0"] = Field(
        default="1.1.0", description="Schema version for compatibility"
    )
    source: SourceInfo
    decision: Decision
    summary: str = Field(
        ...,
        max_length=700,
        description="Opening paragraph + bullet points that give the shape of the video before the details",
    )
    core_thesis: str = Field(
        ...,
        max_length=280,
        description="The ONE idea a viewer should walk away with (fits in a tweet)",
    )
    key_concepts: list[KeyConcept] = Field(
        ..., min_length=3, max_length=5, description="3-5 key concepts from content"
    )
    caveats: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Blind spots, assumptions not supported, claims that need verification (0-5)",
    )
    open_questions: list[str] = Field(
        ..., min_length=1, max_length=5, description="1-5 questions for deeper inquiry"
    )
    actionable_takeaways: list[str] = Field(
        default_factory=list,
        max_length=7,
        description="0-7 concrete next steps a viewer can apply or share (optional)",
    )
    notable_quotes: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="0-3 verbatim sentences from the transcript worth quoting (optional)",
    )
    metadata: Metadata
