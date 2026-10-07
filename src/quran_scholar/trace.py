"""Human-readable research-run trace ([Agent] message)."""

from __future__ import annotations

import logging
from typing import Any

AGENT_LABELS = {
    "planner": "Planner",
    "research_manager": "ResearchManager",
    "quran_researcher": "QuranResearcher",
    "tafsir_researcher": "TafsirResearcher",
    "linguistic_researcher": "LinguisticResearcher",
    "context_researcher": "ContextResearcher",
    "gap_analyzer": "GapAnalyzer",
    "tafsir_comparator": "TafsirComparator",
    "report_generator": "ReportGenerator",
}

_ACTION_PHRASES = {
    "quran_research": "Selecting Quran research...",
    "tafsir_research": "Selecting tafsir research...",
    "linguistic_research": "Selecting linguistic research...",
    "context_research": "Selecting context research...",
    "gap_analysis": "Running gap analysis...",
    "comparison": "Selecting tafsir comparison...",
    "finish": "Proceeding to final answer...",
}

logger = logging.getLogger("quran_scholar.trace")


def _ensure_trace_logging() -> None:
    """One-line trace output on stderr (same feel as former print)."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def trace(agent: str, message: str, *, blank_before: bool = False) -> str:
    _ensure_trace_logging()
    line = f"[{AGENT_LABELS.get(agent, agent)}] {message}"
    if blank_before:
        logger.info("")
    logger.info(line)
    return line


def trace_lines(*lines: str) -> dict[str, Any]:
    return {"trace_log": [ln for ln in lines if ln]}


def manager_trace_message(decision: Any) -> str:
    dispatches = getattr(decision, "dispatches", None) or []
    if len(dispatches) > 1:
        labels = {
            "quran_research": "Quran",
            "tafsir_research": "Tafsir",
            "linguistic_research": "Linguistic",
            "context_research": "Context",
        }
        names = [
            labels.get(getattr(d, "action", None), getattr(d, "action", "?"))
            for d in dispatches
        ]
        return f"Dispatching parallel research: {', '.join(names)}..."
    action = getattr(decision, "action", None)
    return _ACTION_PHRASES.get(action, f"Next action: {action}...")


def format_trace_log(lines: list[str] | None) -> str:
    return "\n".join(lines or [])
