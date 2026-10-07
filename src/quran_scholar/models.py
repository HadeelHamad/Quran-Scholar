"""Pydantic models for structured research outputs and evidence store items."""

from __future__ import annotations

from enum import Enum
from typing import Any

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


class ResearchTask(BaseModel):
    """A single planned research step."""

    id: str
    description: str
    kind: str = Field(
        description="e.g. verse_search, tafsir_fetch, linguistic, nuzool, compare"
    )
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    notes: str | None = None


class ResearchPlan(BaseModel):
    """Planner output: decomposed research tasks for the question."""

    question_summary: str
    approach: str = Field(
        default="",
        description="High-level strategy (parallel fetch, compare tafsirs, etc.)",
    )
    tasks: list[ResearchTask] = Field(default_factory=list)
    target_tafsir_sources: list[str] = Field(
        default_factory=lambda: ["saadi", "katheer", "moyassar"],
        description="Tafsir MCP source ids (e.g. tabary, katheer, saadi)",
    )


class VerseRef(BaseModel):
    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1)
    surah_name: str | None = None


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
    """Word/root linguistic analysis from MCP tools."""

    query: str
    root: str | None = None
    analysis: str
    related_verses: list[VerseRef] = Field(default_factory=list)
    source_tool: str = "analyze_word"
    raw: dict[str, Any] = Field(default_factory=dict)


class NuzoolEvidence(BaseModel):
    """Asbab al-nuzool with isnad when available."""

    ref: VerseRef
    reason: str
    isnad: str | None = None
    source_tool: str = "fetch_nuzool_reason"
    raw: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """Analyst synthesis unit grounded in evidence ids."""

    id: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TafsirComparison(BaseModel):
    """Comparison across multiple tafsir sources for one verse."""

    ref: VerseRef
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    summary: str = ""


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
