"""Specialist researcher nodes (Quran / Tafsir / Linguistic / Context)."""

from __future__ import annotations

from quran_scholar.agents.context_researcher_agent import run_context_research
from quran_scholar.agents.linguistic_researcher_agent import run_linguistic_research
from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research
from quran_scholar.state import ResearchState


def quran_researcher_node(state: ResearchState) -> dict:
    """Find verses: search → evaluate → select (discovered ≠ selected)."""
    return run_quran_research(state)


def tafsir_researcher_node(state: ResearchState) -> dict:
    """Fetch raw tafsir for selected_verses; store separately from findings."""
    return run_tafsir_research(state)


def linguistic_researcher_node(state: ResearchState) -> dict:
    """Optional terminology study — skipped unless plan/task requires it."""
    return run_linguistic_research(state)


def context_researcher_node(state: ResearchState) -> dict:
    """Asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR statuses."""
    return run_context_research(state)
