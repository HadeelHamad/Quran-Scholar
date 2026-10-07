"""Gap Analyzer node."""

from __future__ import annotations

from quran_scholar.agents.gap_analyzer import run_gap_analysis
from quran_scholar.state import ResearchState


def gap_analyzer_node(state: ResearchState) -> dict:
    return run_gap_analysis(state)
