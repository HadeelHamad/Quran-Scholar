"""Deterministic routing helpers for the Quran Scholar graph."""

from __future__ import annotations

from quran_scholar.models import (
    RESEARCHER_ACTIONS,
    ResearchDecision,
    ResearchDispatch,
    ResearchTask,
    TaskStatus,
)
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


def _nodes_from_dispatches(dispatches: list[ResearchDispatch]) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for d in dispatches:
        node = ROUTES.get(d.action)
        if node and node not in seen:
            nodes.append(node)
            seen.add(node)
    return nodes


def route_after_research_manager(state: ResearchState) -> str | list[str]:
    """
    Map ResearchDecision through ROUTES.

    When ``dispatches`` lists multiple independent researchers, return a list
    so LangGraph fans out parallel branches. Dependent work stays a single node.
    """
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision):
        if decision.dispatches:
            nodes = _nodes_from_dispatches(decision.dispatches)
            if len(nodes) > 1:
                return nodes
            if len(nodes) == 1:
                return nodes[0]
        return ROUTES.get(decision.action, DEFAULT_ROUTE)
    if isinstance(decision, dict):
        dispatches = decision.get("dispatches") or []
        if dispatches:
            parsed: list[ResearchDispatch] = []
            for item in dispatches:
                if isinstance(item, ResearchDispatch):
                    parsed.append(item)
                elif isinstance(item, dict) and item.get("action") in ROUTES:
                    parsed.append(ResearchDispatch.model_validate(item))
            nodes = _nodes_from_dispatches(parsed)
            if len(nodes) > 1:
                return nodes
            if len(nodes) == 1:
                return nodes[0]
        action = decision.get("action")
        if action in ROUTES:
            return ROUTES[action]
    return DEFAULT_ROUTE


def task_id_for_action(state: ResearchState, action: str) -> str:
    """
    Resolve the plan task id for a researcher in a (parallel) wave.

    Parallel branches share state; each researcher reads its own dispatch entry
    rather than the single ``current_task_id`` scalar.
    """
    decision = state.get("research_decision")
    if isinstance(decision, ResearchDecision) and decision.dispatches:
        for d in decision.dispatches:
            if d.action == action:
                return d.task_id or ""
    if isinstance(decision, dict):
        for item in decision.get("dispatches") or []:
            if isinstance(item, dict) and item.get("action") == action:
                return str(item.get("task_id") or "")
            if isinstance(item, ResearchDispatch) and item.action == action:
                return item.task_id or ""
    # Fallback: primary task only when it matches this researcher
    if isinstance(decision, ResearchDecision):
        if decision.action == action:
            return decision.task_id or ""
    return state.get("current_task_id") or ""


def route_after_gap_analyzer(state: ResearchState) -> str:
    """Insufficient evidence loops to Research Manager; else continue."""
    if state.get("research_complete") and state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    if _pending_tasks(state) or state.get("gap_status") == "insufficient":
        return "research_manager"
    if state.get("gap_status") == "sufficient":
        return "tafsir_comparator"
    return "research_manager"


def verification_route(state: ResearchState) -> str:
    """Deterministic post-verifier routing — verifier never chooses the route."""
    result = state.get("verification_result")
    if result is not None and getattr(result, "passed", False):
        return "report_generator"

    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    if iteration >= max_iters:
        return "report_generator"

    return "gap_analyzer"


# Alias kept for graph builder imports
route_after_evidence_verifier = verification_route

__all__ = [
    "ROUTES",
    "DEFAULT_ROUTE",
    "RESEARCHER_ACTIONS",
    "route_after_research_manager",
    "route_after_gap_analyzer",
    "verification_route",
    "route_after_evidence_verifier",
    "task_id_for_action",
]
