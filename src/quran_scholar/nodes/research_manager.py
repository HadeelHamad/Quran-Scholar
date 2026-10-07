"""Research Manager — chooses execution pattern and next wave (parallel or sequential)."""

from __future__ import annotations

from quran_scholar.agents.research_manager_agent import (
    choose_execution_pattern,
    decide_next_action,
)
from quran_scholar.state import ResearchState


def research_manager_node(state: ResearchState) -> dict:
    """Emit ResearchDecision (with optional parallel dispatches) + pattern."""
    plan = state.get("research_plan")
    if plan is None:
        return {
            "current_task_id": "",
            "research_decision": None,
            "execution_pattern": None,
            "warnings": ["research_manager: missing research_plan"],
            "errors": ["research_manager: cannot decide without research_plan"],
        }

    pattern = choose_execution_pattern(plan)
    decision = decide_next_action(state)
    if decision.execution_pattern is None:
        decision = decision.model_copy(update={"execution_pattern": pattern})

    return {
        "research_decision": decision,
        "execution_pattern": decision.execution_pattern or pattern,
        "current_task_id": decision.task_id or "",
        "warnings": [
            f"research_manager: pattern={decision.execution_pattern} "
            f"action={decision.action} "
            f"dispatches={[d.action for d in decision.dispatches]}"
        ],
    }
