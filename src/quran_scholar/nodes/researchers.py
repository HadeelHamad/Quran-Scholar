"""Specialist researcher nodes (Quran / Tafsir wired; others still stubs)."""

from __future__ import annotations

from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research
from quran_scholar.state import ResearchState


def _complete_current_task(state: ResearchState, label: str) -> dict:
    task_id = state.get("current_task_id") or ""
    updates: dict = {
        "warnings": [f"{label}: stub completed task {task_id or '(none)'}"],
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    return updates


def quran_researcher_node(state: ResearchState) -> dict:
    """Find verses: search → evaluate → select (discovered ≠ selected)."""
    return run_quran_research(state)


def tafsir_researcher_node(state: ResearchState) -> dict:
    """Fetch raw tafsir for selected_verses; store separately from findings."""
    return run_tafsir_research(state)


def linguistic_researcher_node(state: ResearchState) -> dict:
    """Linguistic Researcher — analyze_word / root tools via Tafsir MCP."""
    return _complete_current_task(state, "linguistic_researcher")


def context_researcher_node(state: ResearchState) -> dict:
    """Context Researcher — nuzool / surah info via Tafsir MCP."""
    return _complete_current_task(state, "context_researcher")
