"""Deterministic routing helpers for the Quran Scholar graph."""

from __future__ import annotations

from quran_scholar.models import ResearchDecision, ResearchTask, TaskStatus
from quran_scholar.state import ResearchState

# Fixed enum → graph node names (LLM never invents node names)
ROUTES = {
    "quran_research": "quran_researcher",
    "tafsir_research": "tafsir_researcher",
    "linguistic_research": "linguistic_researcher",
    "context_research": "context_researcher",
    "gap_analysis": "gap_analyzer",
    "comparison": "tafsir_comparator",
    "verification": "evidence_verifier",
    "finish": "report_generator",
}

DEFAULT_ROUTE = "gap_analyzer"


def _tasks(state: ResearchState) -> list[ResearchTask]:
    plan = state.get("research_plan")
    if plan is None:
        return []
    return list(plan.tasks)


def _pending_tasks(state: ResearchState) -> list[ResearchTask]:
    done = set(state.get("completed_task_ids") or [])
    return [t for t in _tasks(state) if t.id not in done and t.status != TaskStatus.SKIPPED]


def route_after_research_manager(state: ResearchState) -> str:
    """Map ResearchDecision.action through ROUTES (never free-form node names)."""
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision):
        return ROUTES.get(decision.action, DEFAULT_ROUTE)
    if isinstance(decision, dict):
        action = decision.get("action")
        if action in ROUTES:
            return ROUTES[action]
    return DEFAULT_ROUTE


def route_after_gap_analyzer(state: ResearchState) -> str:
    """Insufficient evidence loops to Research Manager; else continue."""
    if state.get("research_complete") and state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    if _pending_tasks(state) or state.get("gap_status") == "insufficient":
        return "research_manager"
    if state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    return "research_manager"


def route_after_evidence_verifier(state: ResearchState) -> str:
    """Failed verification retries via Research Manager; pass → report."""
    if state.get("verification_passed"):
        return "report_generator"

    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    if iteration < max_iters:
        return "research_manager"
    return "report_generator"
