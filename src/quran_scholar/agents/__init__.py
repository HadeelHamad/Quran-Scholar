"""LLM-backed agents (planning, research decisions, analysis, verification)."""

from quran_scholar.agents.llm import get_llm
from quran_scholar.agents.planner_agent import plan_research

__all__ = ["get_llm", "plan_research"]
