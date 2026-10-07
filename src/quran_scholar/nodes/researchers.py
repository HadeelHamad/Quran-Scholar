"""Specialist researcher nodes."""

from __future__ import annotations

from quran_scholar.agents.context_researcher_agent import run_context_research
from quran_scholar.agents.linguistic_researcher_agent import run_linguistic_research
from quran_scholar.agents.quran_researcher_agent import run_quran_research
from quran_scholar.agents.tafsir_researcher_agent import run_tafsir_research
from quran_scholar.nodes.routing import task_id_for_action
from quran_scholar.state import ResearchState

_RUNNERS = {
    "quran_research": run_quran_research,
    "tafsir_research": run_tafsir_research,
    "linguistic_research": run_linguistic_research,
    "context_research": run_context_research,
}


def _run(state: ResearchState, action: str) -> dict:
    patched = {**state, "current_task_id": task_id_for_action(state, action)}
    return _RUNNERS[action](patched)  # type: ignore[arg-type]


def quran_researcher_node(state: ResearchState) -> dict:
    return _run(state, "quran_research")


def tafsir_researcher_node(state: ResearchState) -> dict:
    return _run(state, "tafsir_research")


def linguistic_researcher_node(state: ResearchState) -> dict:
    return _run(state, "linguistic_research")


def context_researcher_node(state: ResearchState) -> dict:
    return _run(state, "context_research")
