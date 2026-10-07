"""Planner node — structured research plan from user question."""

from __future__ import annotations

from quran_scholar.agents.planner import plan_research
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines


def planner_node(state: ResearchState) -> dict:
    """Convert user_question into research_plan (no answer content)."""
    question = (state.get("user_question") or "").strip()
    language = state.get("language") or "ar"

    if not question:
        return {
            "errors": ["planner: empty user_question"],
            "research_iteration": 0,
            **trace_lines(trace("planner", "Empty question — cannot plan.")),
        }

    t0 = trace("planner", "Creating research plan...")
    plan = plan_research(question, language)
    t1 = trace(
        "planner",
        f"Plan ready ({plan.question_focus.value}): "
        f"{len(plan.tasks)} task(s).",
    )
    return {
        "research_plan": plan,
        "research_iteration": 0,
        **trace_lines(t0, t1),
    }
