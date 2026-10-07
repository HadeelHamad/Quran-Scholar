"""LLM-backed agents (planning, research decisions, analysis, verification)."""

from quran_scholar.agents.llm import get_llm
from quran_scholar.agents.planner_agent import plan_research
from quran_scholar.agents.research_manager_agent import decide_next_action

__all__ = ["get_llm", "plan_research", "decide_next_action"]
