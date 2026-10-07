"""Gap Analyzer node — deterministic checks + limited semantic LLM."""

from __future__ import annotations

from quran_scholar.agents.gap_analyzer_agent import (
    analyze_research_gaps,
    apply_gap_to_state_updates,
)
from quran_scholar.state import ResearchState


def gap_analyzer_node(state: ResearchState) -> dict:
    """Decide whether current evidence can answer the question."""
    gap = analyze_research_gaps(state)
    return apply_gap_to_state_updates(state, gap)
