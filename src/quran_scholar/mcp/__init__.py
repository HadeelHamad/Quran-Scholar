"""Tafsir MCP integration layer."""

from quran_scholar.mcp.client import ScopedTafsirMCPClient, TafsirMCPClient, TafsirMCPError
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.safe import MCPCallResult, mark_empty, safe_call_tool
from quran_scholar.mcp.toolsets import TOOLSETS, tools_for

__all__ = [
    "TafsirMCPClient",
    "ScopedTafsirMCPClient",
    "TafsirMCPError",
    "MCPError",
    "MCPCallResult",
    "safe_call_tool",
    "mark_empty",
    "TOOLSETS",
    "tools_for",
]
