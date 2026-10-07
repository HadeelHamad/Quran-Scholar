"""Graph routing: ResearchDecision → node name(s)."""

from __future__ import annotations

from quran_scholar.models import ResearchDecision, TaskStatus
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


def route_after_research_manager(state: ResearchState) -> str | list[str]:
    decision = state.get("research_decision")
    if not isinstance(decision, ResearchDecision):
        return "gap_analyzer"
    if decision.dispatches:
        nodes: list[str] = []
        for d in decision.dispatches:
            node = ROUTES.get(d.action)
            if node and node not in nodes:
                nodes.append(node)
        if len(nodes) > 1:
            return nodes
        if len(nodes) == 1:
            return nodes[0]
    return ROUTES.get(decision.action, "gap_analyzer")


def task_id_for_action(state: ResearchState, action: str) -> str:
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision):
        for d in decision.dispatches:
            if d.action == action:
                return d.task_id or ""
        if decision.action == action:
            return decision.task_id or ""
    return state.get("current_task_id") or ""


def _wants_tafsir_comparison(state: ResearchState) -> bool:
    """Comparator is optional — only when the plan/user question needs it."""
    plan = state.get("research_plan")
    if plan is None:
        return False
    if plan.needs_tafsir_comparison:
        return True
    types = plan.required_evidence_types or []
    return "tafsir_comparison" in types


def route_after_gap_analyzer(state: ResearchState) -> str:
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    pending = bool(
        plan
        and any(
            t.id not in done and t.status != TaskStatus.SKIPPED for t in plan.tasks
        )
    )
    if pending or state.get("gap_status") == "insufficient":
        return "research_manager"
    if state.get("gap_status") == "sufficient":
        if _wants_tafsir_comparison(state):
            return "tafsir_comparator"
        return "claim_extractor"
    return "research_manager"


def verification_route(state: ResearchState) -> str:
    result = state.get("verification_result")
    if result is not None and getattr(result, "passed", False):
        return "report_generator"
    if int(state.get("research_iteration") or 0) >= int(
        state.get("max_research_iterations") or 3
    ):
        return "report_generator"
    return "gap_analyzer"


route_after_evidence_verifier = verification_route
