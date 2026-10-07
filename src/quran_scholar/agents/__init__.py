"""LLM-backed agents (planning, research decisions, analysis, verification)."""

from quran_scholar.agents.context_researcher_agent import run_context_research
from quran_scholar.agents.gap_analyzer_agent import analyze_research_gaps
from quran_scholar.agents.linguistic_researcher_agent import run_linguistic_research
from quran_scholar.agents.llm import get_llm
from quran_scholar.agents.planner_agent import plan_research
from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.research_manager_agent import decide_next_action
from quran_scholar.agents.tafsir_comparator_agent import run_tafsir_comparison
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research

__all__ = [
    "get_llm",
    "plan_research",
    "decide_next_action",
    "run_quran_research",
    "run_tafsir_research",
    "run_linguistic_research",
    "run_context_research",
    "analyze_research_gaps",
    "run_tafsir_comparison",
]

