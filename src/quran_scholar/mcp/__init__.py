"""Tafsir MCP integration layer."""

from quran_scholar.mcp.client import (
    ScopedTafsirMCPClient,
    TafsirMCPClient,
    TafsirMCPError,
    get_tafsir_mcp_tools,
)
from quran_scholar.mcp.errors import (
    MCPError,
    MCPNetworkError,
    MCPProtocolError,
    MCPToolError,
)
from quran_scholar.mcp.safe import MCPCallResult, mark_empty, safe_call_tool
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
    "MCPError",
    "MCPNetworkError",
    "MCPProtocolError",
    "MCPToolError",
    "MCPCallResult",
    "safe_call_tool",
    "mark_empty",
    "get_tafsir_mcp_tools",
    "TOOLSETS",
    "ALL_PROJECT_TOOLS",
    "ToolNotAllowedError",
    "tools_for",
]
