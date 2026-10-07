"""LLM modules: ``*_agent`` = create_agent + tools; others = LLM/heuristics without tools."""

from quran_scholar.agents.claim_extractor import run_claim_extraction
from quran_scholar.agents.context_researcher_agent import run_context_research
from quran_scholar.agents.evidence_verifier import run_evidence_verification
from quran_scholar.agents.gap_analyzer import run_gap_analysis
from quran_scholar.agents.linguistic_researcher_agent import run_linguistic_research
from quran_scholar.agents.llm import get_llm
from quran_scholar.agents.planner import plan_research
from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.report_generator import run_report_generation
from quran_scholar.agents.research_manager import decide_next_action
from quran_scholar.agents.tafsir_comparator import run_tafsir_comparison
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research

__all__ = [
    "get_llm",
    "plan_research",
    "decide_next_action",
    "run_quran_research",
    "run_tafsir_research",
    "run_linguistic_research",
    "run_context_research",
    "run_gap_analysis",
    "run_tafsir_comparison",
    "run_claim_extraction",
    "run_evidence_verification",
    "run_report_generation",
]
