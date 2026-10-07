"""MCP error hierarchy — never confuse transport failure with empty evidence."""

from __future__ import annotations


class MCPError(Exception):
    """Base class for Tafsir MCP failures (network, protocol, tool errors)."""


class TafsirMCPError(MCPError):
    """Raised when the Tafsir MCP HTTP session fails."""


class MCPNetworkError(MCPError):
    """HTTP / connection failure talking to the MCP endpoint."""


class MCPProtocolError(MCPError):
    """Malformed MCP response or JSON-RPC error payload."""


class MCPToolError(MCPError):
    """MCP tool executed but returned an error result."""


class ToolNotAllowedError(PermissionError):
    """Raised when a researcher calls an MCP tool outside its toolset."""
