"""Gap Analyzer — decide if more research is needed (deterministic stub)."""

from __future__ import annotations

from quran_scholar.models import TaskStatus
from quran_scholar.state import ResearchState


def gap_analyzer_node(state: ResearchState) -> dict:
    """Mark gap_status sufficient/insufficient and bump iteration on loops."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    tasks = list(plan.tasks) if plan else []
    pending = [
        t
        for t in tasks
        if t.id not in done and t.status != TaskStatus.SKIPPED
    ]

    evidence_count = len(state.get("evidence_items") or [])
    verse_count = len(state.get("discovered_verses") or []) + len(
        state.get("selected_verses") or []
    )
    tafsir_count = len(state.get("tafsir_evidence") or [])

    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)

    # Stub heuristic: enough once planned tasks are done (or budget exhausted).
    # Real node will score claim coverage / missing sources.
    has_pending = bool(pending)
    budget_left = iteration < max_iters

    if has_pending and budget_left:
        return {
            "gap_status": "insufficient",
            "research_complete": False,
            "research_iteration": iteration + 1,
        }

    sufficient = (not has_pending) or evidence_count > 0 or (
        verse_count + tafsir_count
    ) > 0
    # During early stub phase with no MCP fills, treat finished task list as enough
    if not has_pending:
        sufficient = True

    return {
        "gap_status": "sufficient" if sufficient else "insufficient",
        "research_complete": bool(sufficient or not budget_left),
        "research_iteration": iteration if has_pending else iteration,
    }
