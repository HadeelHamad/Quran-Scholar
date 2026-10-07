"""Optional comparison + final answer nodes."""

from __future__ import annotations

from quran_scholar.agents.report_generator import run_report_generation
from quran_scholar.agents.tafsir_comparator import run_tafsir_comparison
from quran_scholar.state import ResearchState


def tafsir_comparator_node(state: ResearchState) -> dict:
    return run_tafsir_comparison(state)


def report_generator_node(state: ResearchState) -> dict:
    return run_report_generation(state)
