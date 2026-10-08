"""Pydantic models for research plans, evidence, and graph decisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
            "fetch_ayah | quran_search | tafsir_fetch | linguistic | nuzool "
            "(maps to graph researchers)"
        )
    )
    depends_on: list[str] = Field(default_factory=list)
    notes: str | None = None


class ResearchPlan(BaseModel):
    """Planner output: investigation plan only — never an answer to the user."""

    question_summary: str = Field(
        description=(
            "What the user is asking in Arabic (one or two sentences; no tafsir content)"
        )
    )
    required_evidence_types: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence needed for THIS question only, e.g. quran_text, tafsir, "
            "linguistic, nuzool, tafsir_comparison, surah_info, statistics. "
            "Do not include tafsir unless interpretation is required."
        ),
    )
    needs_tafsir_comparison: bool = Field(
        default=False,
        description="True only when multi-mufassir comparison is required",
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
        default_factory=list,
        description=(
            "Tafsir MCP source ids when tafsir is needed (e.g. katheer, saadi). "
            "Leave empty when tafsir is not needed."
        ),
    )
    primary_verse: VerseRef | None = Field(
        default=None,
        description="Primary surah:ayah when the question targets a specific verse",
    )


class VerseEvidence(BaseModel):
    """Quranic verse retrieved via MCP (fetch_ayah / search)."""

    ref: VerseRef
    text_uthmani: str
    source_tool: str = "fetch_ayah"
    relevance: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TafsirEvidence(BaseModel):
    """Tafsir excerpt with attribution."""

    ref: VerseRef
    source_id: str = Field(description="MCP tafsir id, e.g. katheer, saadi")
    source_title: str | None = None
    author: str | None = None
    death_year_hijri: str | None = None
    text: str
    source_tool: str = "fetch_tafsir"
    raw: dict[str, Any] = Field(default_factory=dict)


class LinguisticEvidence(BaseModel):
    """Word/root analysis from MCP tools."""

    query: str
    root: str | None = None
    analysis: str
    related_verses: list[VerseRef] = Field(default_factory=list)
    source_tool: str = "analyze_word"
    raw: dict[str, Any] = Field(default_factory=dict)


class NuzoolEvidence(BaseModel):
    """Asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR status."""

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


class TafsirComparison(BaseModel):
    """Optional comparison across retrieved tafsir sources for one verse."""

    verse_reference: str = Field(description="e.g. '2:153'")
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    difference_types: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    ref: VerseRef | None = None
    source_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class ResearchGap(BaseModel):
    """Gap Analyzer output — deterministic sufficiency check."""

    sufficient: bool
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_tasks: list[ResearchTask] = Field(default_factory=list)


class Evidence(BaseModel):
    """Normalized evidence store entry used for citations and the final answer."""

    id: str
    kind: str = Field(description="verse | tafsir | linguistic | nuzool | quran_meta | other")
    content: str
    citation: str
    refs: list[VerseRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


ResearchAction = Literal[
    "quran_research",
    "tafsir_research",
    "linguistic_research",
    "context_research",
    "gap_analysis",
    "comparison",
    "finish",
]

RESEARCHER_ACTIONS: frozenset[str] = frozenset(
    {
        "quran_research",
        "tafsir_research",
        "linguistic_research",
        "context_research",
    }
)


class ResearchDispatch(BaseModel):
    """One researcher invocation in a (possibly parallel) wave."""

    action: ResearchAction
    task_id: str | None = None


class ResearchDecision(BaseModel):
    """Supervisor decision from Research Manager."""

    action: ResearchAction = Field(
        description="Primary next graph action; parallel waves use dispatches"
    )
    task_id: str | None = Field(default=None)
    dispatches: list[ResearchDispatch] = Field(default_factory=list)
    reasoning: str = Field(description="Brief rationale for this routing decision")
