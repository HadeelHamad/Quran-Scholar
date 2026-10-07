"""Specialist researcher nodes (Quran / Tafsir / Linguistic / Context)."""

from __future__ import annotations

from typing import Any

from quran_scholar.agents.context_researcher_agent import run_context_research
from quran_scholar.agents.linguistic_researcher_agent import run_linguistic_research
from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research
from quran_scholar.nodes.routing import task_id_for_action
from quran_scholar.state import ResearchState


def _with_task_id(state: ResearchState, action: str) -> ResearchState:
    """Bind the correct plan task for this researcher in a parallel wave."""
    tid = task_id_for_action(state, action)
    # Shallow copy is enough — researchers only read task id / plan fields
    patched: dict[str, Any] = dict(state)
    patched["current_task_id"] = tid
    return patched  # type: ignore[return-value]


def quran_researcher_node(state: ResearchState) -> dict:
    """Find verses: search → evaluate → select (discovered ≠ selected)."""
    return run_quran_research(_with_task_id(state, "quran_research"))


def tafsir_researcher_node(state: ResearchState) -> dict:
    """Fetch raw tafsir for selected_verses; store separately from findings."""
    return run_tafsir_research(_with_task_id(state, "tafsir_research"))


def linguistic_researcher_node(state: ResearchState) -> dict:
    """Optional terminology study — skipped unless plan/task requires it."""
    return run_linguistic_research(_with_task_id(state, "linguistic_research"))


def context_researcher_node(state: ResearchState) -> dict:
    """Asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR statuses."""
    return run_context_research(_with_task_id(state, "context_research"))
