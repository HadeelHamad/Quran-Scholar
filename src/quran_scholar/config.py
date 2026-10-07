"""Project configuration (Tafsir sources, limits)."""

from __future__ import annotations

import os

# Recommended scholarly defaults (Tafsir MCP ids)
DEFAULT_TAFSIR_SOURCES = [
    "tabary",  # Al-Tabari
    "katheer",  # Ibn Kathir
    "baghawy",  # Al-Baghawi
    "saadi",  # Al-Sa'di
    "moyassar",  # Al-Muyassar
]


def configured_tafsir_sources(plan_sources: list[str] | None = None) -> list[str]:
    """
    Resolve tafsir source ids.

    Priority: research_plan.target_tafsir_sources → TAFSIR_SOURCES env → defaults.
    """
    if plan_sources:
        return list(plan_sources)
    env = os.getenv("TAFSIR_SOURCES", "").strip()
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    return list(DEFAULT_TAFSIR_SOURCES)


def quran_search_limit() -> int:
    try:
        return max(1, min(50, int(os.getenv("QURAN_SEARCH_LIMIT", "15"))))
    except ValueError:
        return 15
