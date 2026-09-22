"""Pydantic + TypedDict schemas shared across pipeline nodes."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# JD parsing
# ---------------------------------------------------------------------------

class Requirement(BaseModel):
    """A single requirement extracted from the JD. Category is free text —
    the LLM decides what categories exist for this particular JD."""
    category: str = Field(description="Free-text label the LLM assigns, e.g. 'Technical Skills', 'Experience', 'Education'")
    description: str = Field(description="The specific requirement, in the JD's own terms")
    importance: Literal["required", "preferred"] = Field(description="Whether the JD treats this as a must-have or a nice-to-have")


class JDParsed(BaseModel):
    """Structured output of the JD parsing step."""
    job_title: Optional[str] = None
    requirements: list[Requirement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class RequirementVerdict(BaseModel):
    """The matcher's judgment on one requirement for one candidate."""
    category: str
    description: str
    importance: Literal["required", "preferred"]
    verdict: Literal["matched", "partial", "missing"]
    evidence: Optional[str] = Field(default=None, description="Short quote/paraphrase from the CV supporting the verdict")


class CandidateMatchResult(BaseModel):
    candidate_id: str
    verdicts: list[RequirementVerdict] = Field(default_factory=list)


class RequirementJudgement(BaseModel):
    """Lightweight per-requirement verdict from the matcher LLM — references
    the requirement by index rather than re-typing its text, to keep output
    tokens down and avoid the model drifting from the original wording."""
    requirement_index: int = Field(description="0-based index into the numbered requirements list given in the prompt")
    verdict: Literal["matched", "partial", "missing"]
    evidence: Optional[str] = Field(default=None, description="Short quote/paraphrase (<20 words) from the CV. Null if missing.")


class MatchOutput(BaseModel):
    judgements: list[RequirementJudgement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class CandidateScore(BaseModel):
    candidate_id: str
    score: float
    status: Literal["Shortlisted", "Needs Review", "Not shortlisted"]
    review_reason: Optional[str] = Field(default=None, description="Why this candidate needs a human look, if applicable")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class Candidate(TypedDict, total=False):
    candidate_id: str
    file_path: str  # path to the raw CV file (PDF/DOCX/etc.) — omit if raw_text is given
    raw_text: str   # optional: skip file conversion entirely and use this text as-is


class ScreeningState(TypedDict, total=False):
    # Inputs
    jd_raw_text: str        # provide this OR jd_file_path
    jd_file_path: str       # optional: JD as a PDF/DOCX file, converted via MarkItDown
    candidates: list[Candidate]
    threshold: float  # e.g. 70.0
    review_margin: float  # e.g. 5.0 — score within threshold +/- this band -> Needs Review

    # Intermediate / outputs
    jd_parsed: JDParsed
    cv_texts: dict[str, str]                        # candidate_id -> cleaned Markdown/text
    match_results: dict[str, CandidateMatchResult]   # candidate_id -> match result
    scores: dict[str, CandidateScore]                # candidate_id -> score + status
    report: list[dict]                               # final flattened rows for display
