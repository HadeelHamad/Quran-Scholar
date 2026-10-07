"""Shared LangGraph research state.

Nodes return only the fields they change — never a full reconstructed state.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from quran_scholar.models import (
    Claim,
    Evidence,
    Finding,
    LinguisticEvidence,
    NuzoolEvidence,
    ResearchPlan,
    TafsirComparison,
    TafsirEvidence,
    VerificationResult,
    VerseEvidence,
)


def _extend_list(left: list | None, right: list | None) -> list:
    """Reducer: append new items (used for warnings/errors accumulators)."""
    return (left or []) + (right or [])


class ResearchState(TypedDict, total=False):
    """Single shared graph state for Quran Scholar research."""

    # Input
    user_question: str
    language: str

    # Planning
    research_plan: ResearchPlan
    current_task_id: str
    completed_task_ids: list[str]

    # Quran research
    discovered_verses: list[VerseEvidence]
    selected_verses: list[VerseEvidence]

    # Tafsir research
    tafsir_evidence: list[TafsirEvidence]

    # Supporting research
    linguistic_evidence: list[LinguisticEvidence]
    nuzool_evidence: list[NuzoolEvidence]

    # Analysis
    findings: list[Finding]
    tafsir_comparisons: list[TafsirComparison]

    # Evidence / claims
    evidence_items: list[Evidence]
    claims: list[Claim]

    # Verification
    verification_result: VerificationResult | None
    unsupported_claims: list[Claim]

    # Control
    research_iteration: int
    max_research_iterations: int
    research_complete: bool
    verification_passed: bool

    # Output
    final_report: str | None

    # Diagnostics (append-only via reducer when used with Annotated)
    warnings: Annotated[list[str], _extend_list]
    errors: Annotated[list[str], _extend_list]

    # Optional agent message channel for LLM nodes
    messages: Annotated[list, add_messages]


def initial_research_state(
    user_question: str,
    *,
    language: str = "ar",
    max_research_iterations: int = 3,
) -> ResearchState:
    """Build the starting state for a research run."""
    return ResearchState(
        user_question=user_question,
        language=language,
        completed_task_ids=[],
        discovered_verses=[],
        selected_verses=[],
        tafsir_evidence=[],
        linguistic_evidence=[],
        nuzool_evidence=[],
        findings=[],
        tafsir_comparisons=[],
        evidence_items=[],
        claims=[],
        unsupported_claims=[],
        verification_result=None,
        research_iteration=0,
        max_research_iterations=max_research_iterations,
        research_complete=False,
        verification_passed=False,
        final_report=None,
        warnings=[],
        errors=[],
        messages=[],
    )
