"""Planner node — structured research plan from user question."""

from __future__ import annotations

from quran_scholar.agents.planner import plan_research
from quran_scholar.state import ResearchState


def planner_node(state: ResearchState) -> dict:
    """Convert user_question into research_plan (no answer content)."""
    question = (state.get("user_question") or "").strip()
    language = state.get("language") or "ar"

    if not question:
        return {
            "errors": ["planner: empty user_question"],
            "research_iteration": 0,
        }

    plan = plan_research(question, language)
    return {
        "research_plan": plan,
        "research_iteration": 0,
    }
