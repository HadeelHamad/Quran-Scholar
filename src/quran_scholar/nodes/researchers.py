"""Specialist researcher nodes (stubs — MCP wiring comes next)."""

from __future__ import annotations

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
    """Quran Researcher — verse search / fetch_ayah via Tafsir MCP."""
    return _complete_current_task(state, "quran_researcher")


def tafsir_researcher_node(state: ResearchState) -> dict:
    """Tafsir Researcher — fetch_tafsir / search_in_tafsir via Tafsir MCP."""
    return _complete_current_task(state, "tafsir_researcher")


def linguistic_researcher_node(state: ResearchState) -> dict:
    """Linguistic Researcher — analyze_word / root tools via Tafsir MCP."""
    return _complete_current_task(state, "linguistic_researcher")


def context_researcher_node(state: ResearchState) -> dict:
    """Context Researcher — nuzool / surah info via Tafsir MCP."""
    return _complete_current_task(state, "context_researcher")
