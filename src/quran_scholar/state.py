"""Shared LangGraph research state.

Nodes return only the fields they change — never a full reconstructed state.

Collections use append-style reducers (``operator.add``) so later research
iterations cannot wipe earlier evidence. Scalar / "current value" fields use
normal replacement semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from quran_scholar.models import (
    Claim,
    Evidence,
    ExecutionPattern,
    Finding,
    LinguisticEvidence,
    NuzoolEvidence,
    ResearchDecision,
    ResearchGap,
    ResearchPlan,
    TafsirComparison,
    TafsirEvidence,
    VerificationResult,
    VerseEvidence,
)


class ResearchState(TypedDict, total=False):
    """Single shared graph state for Quran Scholar research."""

    # Input (replace)
    user_question: str
    language: str

    # Planning / supervision
    research_plan: ResearchPlan  # replace
    research_decision: ResearchDecision | None  # replace — supervisor output
    execution_pattern: ExecutionPattern | None  # replace — thematic / verse / comparison
    current_task_id: str  # replace — primary task; parallel waves use decision.dispatches
    completed_task_ids: Annotated[list[str], operator.add]  # append
    unresolved_gaps: list[str]  # replace — current gap list from gap analyzer
    research_gap: ResearchGap | None  # replace — full Gap Analyzer output

    # Quran research (append — accumulate across iterations)
    discovered_verses: Annotated[list[VerseEvidence], operator.add]
    selected_verses: Annotated[list[VerseEvidence], operator.add]

    # Tafsir research (append)
    tafsir_evidence: Annotated[list[TafsirEvidence], operator.add]

    # Supporting research (append)
    linguistic_evidence: Annotated[list[LinguisticEvidence], operator.add]
    nuzool_evidence: Annotated[list[NuzoolEvidence], operator.add]

    # Analysis (append)
    findings: Annotated[list[Finding], operator.add]
    tafsir_comparisons: Annotated[list[TafsirComparison], operator.add]

    # Evidence (append) / claims (replace — extractor + verifier rewrite statuses)
    evidence_items: Annotated[list[Evidence], operator.add]
    claims: list[Claim]

    # Verification (replace — current snapshot)
    verification_result: VerificationResult | None
    unsupported_claims: list[Claim]

    # Control (replace)
    research_iteration: int
    max_research_iterations: int
    research_complete: bool
    verification_passed: bool
    gap_status: str  # "insufficient" | "sufficient"

    # Output (replace)
    final_report: str | None

    # Diagnostics (append)
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

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
        gap_status="insufficient",
        final_report=None,
        warnings=[],
        errors=[],
        messages=[],
    )
