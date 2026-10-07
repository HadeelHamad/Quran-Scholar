"""Research Manager node — pattern + next wave."""

from __future__ import annotations

from quran_scholar.agents.research_manager import (
    choose_execution_pattern,
    decide_next_action,
)
from quran_scholar.state import ResearchState
from quran_scholar.trace import manager_trace_message, trace, trace_lines


def research_manager_node(state: ResearchState) -> dict:
    plan = state.get("research_plan")
    if plan is None:
        return {
            "current_task_id": "",
            "research_decision": None,
            "execution_pattern": None,
            "warnings": ["research_manager: missing research_plan"],
            "errors": ["research_manager: cannot decide without research_plan"],
            **trace_lines(
                trace("research_manager", "Missing research plan — cannot route.")
            ),
        }

    pattern = choose_execution_pattern(plan)
    decision = decide_next_action(state)
    if decision.execution_pattern is None:
        decision = decision.model_copy(update={"execution_pattern": pattern})

    return {
        "research_decision": decision,
        "execution_pattern": decision.execution_pattern or pattern,
        "current_task_id": decision.task_id or "",
        **trace_lines(trace("research_manager", manager_trace_message(decision))),
    }
