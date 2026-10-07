"""Tafsir MCP integration layer."""

from quran_scholar.mcp.client import (
    ScopedTafsirMCPClient,
    TafsirMCPClient,
    TafsirMCPError,
    get_tafsir_mcp_tools,
)
from quran_scholar.mcp.toolsets import (
    ALL_PROJECT_TOOLS,
    TOOLSETS,
    ToolNotAllowedError,
    tools_for,
)

__all__ = [
    "TafsirMCPClient",
    "ScopedTafsirMCPClient",
    "TafsirMCPError",
    "get_tafsir_mcp_tools",
    "TOOLSETS",
    "ALL_PROJECT_TOOLS",
    "ToolNotAllowedError",
    "tools_for",
]
