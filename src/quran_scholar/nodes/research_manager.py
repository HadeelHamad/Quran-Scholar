"""Research Manager — supervisor that chooses the next fixed action."""

from __future__ import annotations

from quran_scholar.agents.research_manager_agent import decide_next_action
from quran_scholar.state import ResearchState


def research_manager_node(state: ResearchState) -> dict:
    """Emit ResearchDecision and current_task_id for deterministic routing."""
    plan = state.get("research_plan")
    if plan is None:
        return {
            "current_task_id": "",
            "research_decision": None,
            "warnings": ["research_manager: missing research_plan"],
            "errors": ["research_manager: cannot decide without research_plan"],
        }

    decision = decide_next_action(state)
    return {
        "research_decision": decision,
        "current_task_id": decision.task_id or "",
    }
