"""
schemas.py — Pydantic-validated JSON contracts for LLM output.

Every synthesis/report call asks the model for JSON matching one of these
schemas, then validates the response through Pydantic rather than trusting
raw dict.get() calls. A ValidationError is treated exactly like any other
synthesis failure (clean fallback message, no extra LLM call) — never a
reason to retry the SAME model or invent data to satisfy the schema.

Fields use permissive defaults (empty list/None) rather than requiring every
field, because the model must be free to say "not available in the uploaded
evidence" instead of being forced to hallucinate something to pass
validation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Optional


def _coerce_to_str_list(v):
    """LLMs occasionally return a single string instead of a one-item list
    for these fields — coerce rather than fail validation over formatting."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(item) for item in v if str(item).strip()]
    return []


# ---------------------------------------------------------------------------
# Normal answer — compact schema (item 2)
# ---------------------------------------------------------------------------

class AnswerJSON(BaseModel):
    """The ONE synthesis call's structured output for a normal question."""
    answer: str = Field(..., description="Concise, grounded, natural-language answer.")
    grounded: bool = Field(True, description="True only if every statement traces to the evidence.")
    evidence_sufficient: bool = Field(True, description="True only if the evidence actually answers the question.")
    document_ids: list[str] = Field(default_factory=list, description="document_ids the answer actually drew on.")

    @field_validator("document_ids", mode="before")
    @classmethod
    def _coerce_document_ids(cls, v):
        return _coerce_to_str_list(v)


# ---------------------------------------------------------------------------
# Structured research extraction (Feature 6) — same evidence, same ONE
# synthesis call as AnswerJSON, just a richer requested shape. Only used
# when a request explicitly opts into it (mode="structured"); normal
# conversational answers never pay the extra output tokens for this.
# ---------------------------------------------------------------------------

class StructuredFindings(BaseModel):
    architecture: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    methodology: Optional[str] = None
    key_calculations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    final_summary: Optional[str] = None

    @field_validator("datasets", "metrics", "contributions", "key_calculations", "limitations", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return _coerce_to_str_list(v)


class StructuredAnswerJSON(BaseModel):
    """AnswerJSON's structured sibling — validated the same way, same
    citation-gating rule applies (citations only attach on real success)."""
    answer: str = Field(..., description="Concise, grounded, natural-language answer.")
    grounded: bool = True
    evidence_sufficient: bool = True
    document_ids: list[str] = Field(default_factory=list)
    structured: StructuredFindings

    @field_validator("document_ids", mode="before")
    @classmethod
    def _coerce_document_ids(cls, v):
        return _coerce_to_str_list(v)


# ---------------------------------------------------------------------------
# Report generation (item 7)
# ---------------------------------------------------------------------------

class PaperReport(BaseModel):
    document_id: str
    title: Optional[str] = None
    overview: Optional[str] = None
    main_contribution: Optional[str] = None
    methodology: Optional[str] = None
    architecture: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    key_results: list[str] = Field(default_factory=list)
    important_calculations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    final_summary: Optional[str] = None

    @field_validator("datasets", "evaluation_metrics", "key_results", "important_calculations", "limitations", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return _coerce_to_str_list(v)


class ComparisonReport(BaseModel):
    commonalities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("commonalities", "differences", "strengths", "limitations", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return _coerce_to_str_list(v)


class ResearchReport(BaseModel):
    title: str
    overview: str
    papers: list[PaperReport] = Field(default_factory=list)
    comparison: Optional[ComparisonReport] = None
    conclusion: str
    evidence_sufficient: bool = True


# ---------------------------------------------------------------------------
# Interview / analysis modes (Viva, Mock Test, Project Interview) — still ONE
# LLM call per request, same validate-don't-trust discipline as everything
# above.
# ---------------------------------------------------------------------------

class InterviewQuestion(BaseModel):
    question: str
    category: str = ""
    difficulty: str = "intermediate"
    expected_answer_points: list[str] = Field(default_factory=list)

    @field_validator("expected_answer_points", mode="before")
    @classmethod
    def _coerce_points(cls, v):
        return _coerce_to_str_list(v)


class QuestionSetJSON(BaseModel):
    """Output of ONE call that generates a batch of grounded questions
    (Viva / Mock Test / Project Interview opening question set)."""
    questions: list[InterviewQuestion] = Field(default_factory=list)
    grounded: bool = True
    evidence_sufficient: bool = True

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class AnswerEvaluationJSON(BaseModel):
    """Output of ONE call that evaluates a user's answer during Project
    Interview mode and proposes the next question."""
    correctness: str = Field("unclear", description="'correct' | 'partially_correct' | 'incorrect' | 'unclear'")
    missing_points: list[str] = Field(default_factory=list)
    technical_depth: str = Field("", description="Brief assessment of depth/precision shown.")
    suggested_answer: str = Field("", description="A better/more complete answer, grounded in the real implementation.")
    next_question: str = Field("", description="The next interview question, or empty if the interview is complete.")
    next_question_category: str = ""

    @field_validator("missing_points", mode="before")
    @classmethod
    def _coerce_missing(cls, v):
        return _coerce_to_str_list(v)
