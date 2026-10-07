"""Tafsir MCP integration layer."""

from quran_scholar.mcp.client import (
    TafsirMCPClient,
    TafsirMCPError,
    get_tafsir_mcp_tools,
)

__all__ = [
    "TafsirMCPClient",
    "TafsirMCPError",
    "get_tafsir_mcp_tools",
]
