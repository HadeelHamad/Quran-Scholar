"""Graph routing: ResearchDecision → node name(s)."""

from __future__ import annotations

from quran_scholar.models import ResearchDecision
from quran_scholar.state import ResearchState

ROUTES = {
    "quran_research": "quran_researcher",
    "tafsir_research": "tafsir_researcher",
    "linguistic_research": "linguistic_researcher",
    "context_research": "context_researcher",
    "gap_analysis": "gap_analyzer",
    "comparison": "tafsir_comparator",
    "finish": "report_generator",
}


def route_after_research_manager(state: ResearchState) -> str | list[str]:
    """Only the Research Manager chooses the next node(s)."""
    decision = state.get("research_decision")
    if not isinstance(decision, ResearchDecision):
        # Safe default: never bounce to gap without a decision (recursion risk).
        return "report_generator"
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
    return ROUTES.get(decision.action, "report_generator")


def task_id_for_action(state: ResearchState, action: str) -> str:
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision):
        for d in decision.dispatches:
            if d.action == action:
                return d.task_id or ""
        if decision.action == action:
            return decision.task_id or ""
    return state.get("current_task_id") or ""
