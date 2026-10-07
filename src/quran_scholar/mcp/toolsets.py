"""Focused MCP toolsets per researcher — LLM chooses among allowed tools."""

from __future__ import annotations

from typing import Literal

ResearcherRole = Literal[
    "quran",
    "tafsir",
    "linguistic",
    "context",
]

# Broad Quran research: each role gets the MCP tools relevant to its specialty
TOOLSETS: dict[ResearcherRole, frozenset[str]] = {
    "quran": frozenset(
        {
            "search_quran_text",
            "fetch_ayah",
            "fetch_surah_info",
            "get_quran_overview",
            "get_surah_statistics",
            "get_qeraat_variants",
            "get_page_fawaed",
        }
    ),
    "linguistic": frozenset(
        {
            "analyze_word",
            "find_root_occurrences",
            "get_root_stats",
        }
    ),
    "tafsir": frozenset(
        {
            "fetch_tafsir",
            "search_in_tafsir",
            "list_tafsir_sources",
            "list_sources_for_ayah",
        }
    ),
    "context": frozenset(
        {
            "fetch_nuzool_reason",
            "list_science_sources",
            "list_all_sources",
        }
    ),
}

ALL_PROJECT_TOOLS: frozenset[str] = frozenset().union(*TOOLSETS.values())


class ToolNotAllowedError(PermissionError):
    """Raised when a researcher calls an MCP tool outside its toolset."""


def tools_for(role: ResearcherRole) -> frozenset[str]:
    return TOOLSETS[role]


def assert_tool_allowed(role: ResearcherRole, tool_name: str) -> None:
    allowed = TOOLSETS[role]
    if tool_name not in allowed:
        raise ToolNotAllowedError(
            f"Tool '{tool_name}' is not in the '{role}' toolset. "
            f"Allowed: {sorted(allowed)}"
        )
