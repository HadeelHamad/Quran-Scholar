"""Shared LangGraph research state.

Nodes return only the fields they change — never a full reconstructed state.

Collections use append-style reducers (``operator.add``) so later research
iterations cannot wipe earlier evidence. Scalar fields use replacement semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from quran_scholar.models import (
    Evidence,
    ExecutionPattern,
    LinguisticEvidence,
    NuzoolEvidence,
    ResearchDecision,
    ResearchGap,
    ResearchPlan,
    TafsirComparison,
    TafsirEvidence,
    VerseEvidence,
)


class ResearchState(TypedDict, total=False):
    """Single shared graph state for Quran Scholar research."""

    user_question: str
    language: str

    research_plan: ResearchPlan
    research_decision: ResearchDecision | None
    execution_pattern: ExecutionPattern | None
    current_task_id: str
    completed_task_ids: Annotated[list[str], operator.add]
    unresolved_gaps: list[str]
    research_gap: ResearchGap | None

    discovered_verses: Annotated[list[VerseEvidence], operator.add]
    selected_verses: Annotated[list[VerseEvidence], operator.add]
    tafsir_evidence: Annotated[list[TafsirEvidence], operator.add]
    linguistic_evidence: Annotated[list[LinguisticEvidence], operator.add]
    nuzool_evidence: Annotated[list[NuzoolEvidence], operator.add]
    tafsir_comparisons: Annotated[list[TafsirComparison], operator.add]
    evidence_items: Annotated[list[Evidence], operator.add]

    research_iteration: int
    max_research_iterations: int
    research_complete: bool
    gap_status: str  # "insufficient" | "sufficient"

    final_report: str | None

    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    messages: Annotated[list, add_messages]


def initial_research_state(
    user_question: str,
    *,
    language: str = "ar",
    max_research_iterations: int = 3,
) -> ResearchState:
    """Build the starting state for a research run (Arabic question text)."""
    return ResearchState(
        user_question=user_question,
        language=language,
        research_decision=None,
        execution_pattern=None,
        research_gap=None,
        completed_task_ids=[],
        unresolved_gaps=[],
        discovered_verses=[],
        selected_verses=[],
        tafsir_evidence=[],
        linguistic_evidence=[],
        nuzool_evidence=[],
        tafsir_comparisons=[],
        evidence_items=[],
        research_iteration=0,
        max_research_iterations=max_research_iterations,
        research_complete=False,
        gap_status="insufficient",
        final_report=None,
        warnings=[],
        errors=[],
        messages=[],
    )
