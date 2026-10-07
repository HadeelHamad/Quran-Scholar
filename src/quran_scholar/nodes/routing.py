"""Deterministic routing helpers for the Quran Scholar graph."""

from __future__ import annotations

from quran_scholar.models import ResearchTask, TaskStatus
from quran_scholar.state import ResearchState

RESEARCHER_BY_KIND = {
    "fetch_ayah": "quran_researcher",
    "quran_search": "quran_researcher",
    "verse_search": "quran_researcher",
    "quran": "quran_researcher",
    "tafsir_fetch": "tafsir_researcher",
    "fetch_tafsir": "tafsir_researcher",
    "tafsir_research": "tafsir_researcher",
    "tafsir": "tafsir_researcher",
    "linguistic": "linguistic_researcher",
    "linguistic_analysis": "linguistic_researcher",
    "nuzool": "context_researcher",
    "nuzool_research": "context_researcher",
    "context": "context_researcher",
}


def _tasks(state: ResearchState) -> list[ResearchTask]:
    plan = state.get("research_plan")
    if plan is None:
        return []
    return list(plan.tasks)


def _pending_tasks(state: ResearchState) -> list[ResearchTask]:
    done = set(state.get("completed_task_ids") or [])
    return [t for t in _tasks(state) if t.id not in done and t.status != TaskStatus.SKIPPED]


def route_after_research_manager(state: ResearchState) -> str:
    """Select which specialist researcher handles the current task."""
    task_id = state.get("current_task_id")
    for task in _tasks(state):
        if task.id == task_id:
            return RESEARCHER_BY_KIND.get(task.kind, "quran_researcher")
    # No task selected → skip research and let gap analyzer decide
    return "gap_analyzer"


def route_after_gap_analyzer(state: ResearchState) -> str:
    """Insufficient evidence loops to Research Manager; else continue."""
    if state.get("research_complete"):
        return "tafsir_comparator"
    if _pending_tasks(state):
        return "research_manager"
    if state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    # Still insufficient but no pending tasks — force analysis path
    return "tafsir_comparator"


def route_after_evidence_verifier(state: ResearchState) -> str:
    """Failed verification retries via Gap Analyzer; pass → report."""
    if state.get("verification_passed"):
        return "report_generator"

    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    if iteration < max_iters:
        return "gap_analyzer"
    # Budget exhausted — still produce a report
    return "report_generator"
