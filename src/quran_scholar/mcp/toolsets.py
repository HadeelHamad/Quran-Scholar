"""Focused MCP toolsets per researcher — do not expose every tool to every agent."""

from __future__ import annotations

from typing import Literal

ResearcherRole = Literal[
    "quran",
    "tafsir",
    "linguistic",
    "context",
    "specialized",
]

# Explicit allow-lists (reduces tool-selection errors and token usage)
TOOLSETS: dict[ResearcherRole, frozenset[str]] = {
    "quran": frozenset(
        {
            "search_quran_text",
            "fetch_ayah",
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
        }
    ),
    "context": frozenset(
        {
            "fetch_nuzool_reason",
        }
    ),
    "specialized": frozenset(
        {
            "get_qeraat_variants",
            "fetch_surah_info",
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
