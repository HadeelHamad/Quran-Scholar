"""Pydantic models for structured research outputs and evidence store items."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


class ClaimSupport(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class QuestionFocus(str, Enum):
    VERSE_SPECIFIC = "verse_specific"
    THEMATIC = "thematic"
    MIXED = "mixed"


class VerseRef(BaseModel):
    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1)
    surah_name: str | None = None


class ResearchTask(BaseModel):
    """A single planned research step executed by a specialist researcher."""

    id: str
    description: str
    kind: str = Field(
        description=(
            "Research step kind: fetch_ayah, quran_search, tafsir_fetch, "
            "linguistic, nuzool (maps to graph researchers; do not use "
            "verification or tafsir_comparison as task kinds — those are "
            "downstream graph stages)"
        )
    )
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    notes: str | None = None


class ResearchPlan(BaseModel):
    """Planner output: investigation plan only — never an answer to the user."""

    question_summary: str = Field(
        description="What the user is asking, in one or two sentences (no tafsir content)"
    )
    question_focus: QuestionFocus = Field(
        description="verse_specific if a surah:ayah is central; thematic if topic-based"
    )
    required_evidence_types: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence needed, e.g. quran_text, tafsir, linguistic, nuzool, "
            "tafsir_comparison (comparison is a downstream stage, not a researcher task)"
        ),
    )
    needs_tafsir_comparison: bool = Field(
        default=False,
        description="True when multiple tafsir sources should be compared in analysis",
    )
    needs_linguistic_analysis: bool = Field(
        default=False,
        description="True when root/word analysis adds value",
    )
    needs_sabab_nuzool: bool = Field(
        default=False,
        description="True when asbab al-nuzool context is relevant",
    )
    approach: str = Field(
        default="",
        description="High-level investigation strategy (no Quranic quotations)",
    )
    tasks: list[ResearchTask] = Field(
        default_factory=list,
        description="Minimal set of researcher tasks; avoid unnecessary steps",
    )
    target_tafsir_sources: list[str] = Field(
        default_factory=lambda: ["saadi", "katheer", "moyassar"],
        description="Tafsir MCP source ids when tafsir is needed (e.g. katheer, saadi)",
    )
    primary_verse: VerseRef | None = Field(
        default=None,
        description="Primary surah:ayah when question_focus is verse_specific",
    )


class VerseEvidence(BaseModel):
    """Quranic verse retrieved via Tafsir MCP (e.g. fetch_ayah / search)."""

    ref: VerseRef
    text_uthmani: str
    source_tool: str = "fetch_ayah"
    relevance: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TafsirEvidence(BaseModel):
    """Tafsir excerpt with mandatory attribution."""

    ref: VerseRef
    source_id: str = Field(description="MCP tafsir id, e.g. katheer, saadi")
    source_title: str | None = None
    author: str | None = None
    death_year_hijri: str | None = None
    text: str
    source_tool: str = "fetch_tafsir"
    raw: dict[str, Any] = Field(default_factory=dict)


class LinguisticEvidence(BaseModel):
    """Word/root linguistic analysis from MCP tools (raw tool output, not LLM paraphrase)."""

    query: str
    root: str | None = None
    analysis: str
    related_verses: list[VerseRef] = Field(default_factory=list)
    source_tool: str = "analyze_word"
    raw: dict[str, Any] = Field(default_factory=dict)


class NuzoolStatus(str, Enum):
    """Distinguish missing data from errors — never collapse all to None."""

    FOUND = "FOUND"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    ERROR = "ERROR"


class NuzoolEvidence(BaseModel):
    """Asbab al-nuzool with explicit FOUND / NOT_AVAILABLE / ERROR status."""

    status: Literal["FOUND", "NOT_AVAILABLE", "ERROR"]
    content: str | None = None
    source: str | None = None
    surah_number: int = Field(ge=1, le=114)
    ayah_number: int = Field(ge=1)
    isnad: str | None = None
    source_tool: str = "fetch_nuzool_reason"
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def ref(self) -> VerseRef:
        return VerseRef(surah=self.surah_number, ayah=self.ayah_number)


class Finding(BaseModel):
    """Analyst synthesis unit grounded in evidence ids."""

    id: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TafsirComparison(BaseModel):
    """Comparison across retrieved tafsir sources for one verse (evidence-grounded)."""

    verse_reference: str = Field(
        description="Human-readable verse ref, e.g. '2:153'"
    )
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    difference_types: list[str] = Field(
        default_factory=list,
        description="Categories of real interpretive differences when present",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Ids of tafsir evidence items used in this comparison",
    )
    # Compatibility / structured ref
    ref: VerseRef | None = None
    source_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class ResearchGap(BaseModel):
    """Gap Analyzer output — mostly from deterministic checks."""

    sufficient: bool
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_tasks: list[ResearchTask] = Field(default_factory=list)


class Evidence(BaseModel):
    """Normalized evidence store entry referenced by claims."""

    id: str
    kind: str = Field(
        description="verse | tafsir | linguistic | nuzool | other"
    )
    content: str
    citation: str
    refs: list[VerseRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A claim that must be grounded in evidence_items."""

    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    support: ClaimSupport = ClaimSupport.UNKNOWN
    notes: str | None = None


class VerificationResult(BaseModel):
    """Verifier node structured output."""

    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    missing_evidence_notes: list[str] = Field(default_factory=list)
    needs_more_research: bool = False


ResearchAction = Literal[
    "quran_research",
    "tafsir_research",
    "linguistic_research",
    "context_research",
    "gap_analysis",
    "comparison",
    "verification",
    "finish",
]


class ResearchDecision(BaseModel):
    """Supervisor decision from Research Manager — fixed action enum only."""

    action: ResearchAction = Field(
        description=(
            "Next graph action from the fixed enum only. "
            "Never invent node names."
        )
    )
    task_id: str | None = Field(
        default=None,
        description="Plan task id when dispatching a researcher; else null",
    )
    reasoning: str = Field(
        description="Brief rationale for this routing decision"
    )
