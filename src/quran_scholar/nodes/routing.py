"""Graph routing: ResearchDecision → node name(s)."""

from __future__ import annotations

from quran_scholar.models import ResearchDecision, ResearchDispatch, TaskStatus
from quran_scholar.state import ResearchState

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


def _has_pending_tasks(state: ResearchState) -> bool:
    plan = state.get("research_plan")
    if plan is None:
        return False
    done = set(state.get("completed_task_ids") or [])
    return any(
        t.id not in done and t.status != TaskStatus.SKIPPED for t in plan.tasks
    )


def _nodes_for(dispatches: list[ResearchDispatch]) -> list[str]:
    nodes: list[str] = []
    for d in dispatches:
        node = ROUTES.get(d.action)
        if node and node not in nodes:
            nodes.append(node)
    return nodes


def route_after_research_manager(state: ResearchState) -> str | list[str]:
    """Fan out to a list of nodes when dispatches has multiple researchers."""
    decision = state.get("research_decision")
    if not isinstance(decision, ResearchDecision):
        return "gap_analyzer"

    if decision.dispatches:
        nodes = _nodes_for(decision.dispatches)
        if len(nodes) > 1:
            return nodes
        if len(nodes) == 1:
            return nodes[0]

    return ROUTES.get(decision.action, "gap_analyzer")


def task_id_for_action(state: ResearchState, action: str) -> str:
    """Task id for one researcher in a parallel wave."""
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision):
        for d in decision.dispatches:
            if d.action == action:
                return d.task_id or ""
        if decision.action == action:
            return decision.task_id or ""
    return state.get("current_task_id") or ""


def route_after_gap_analyzer(state: ResearchState) -> str:
    if state.get("gap_status") == "sufficient" and not _has_pending_tasks(state):
        return "tafsir_comparator"
    if _has_pending_tasks(state) or state.get("gap_status") == "insufficient":
        return "research_manager"
    if state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    return "research_manager"


def verification_route(state: ResearchState) -> str:
    result = state.get("verification_result")
    if result is not None and getattr(result, "passed", False):
        return "report_generator"
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    if iteration >= max_iters:
        return "report_generator"
    return "gap_analyzer"


route_after_evidence_verifier = verification_route
